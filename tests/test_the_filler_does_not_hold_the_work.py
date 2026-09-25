"""つなぎの声が鳴り終わるのを、調べものと主LLM が待たない（出-aq 段 1・2026-09-24）。

つなぎは「**時間がかかっているあいだ、無視していないと伝える**」ためにある（本人）。
ところが実機 2026-09-24 15:58、検索はつなぎが鳴り終わるまで **4.59 秒**待たされていた。

    15:58:45.4  調停が「調べる」と決め、つなぎ「明日の天気ですね。調べてみます。」を返す
                ├ つなぎを声にする ………… 4.59 秒待つ
    15:58:49.96 ├ 声が鳴り終わる → ここで初めて画面に吹き出しが出る
                └ ここで初めて検索を投げる

`_say_filler` が `await self._dif.speak(text)` で**再生の終わりまで待ってから**戻っていた
ためである。口 2（`full`）でも主LLM は同じく待たされる。**待ちを埋めるはずのつなぎが、
待ちを延ばしていた。**

直すのは、**声だけを背景に回す**ことである。画面・記録・O への書き込みはすぐやる
（相手が聞くのとパジュが読み返すのとで、食い違わせない）。声が本応答と重ならないのは、
`TTSTool` の鍵（一度に一つしか鳴らさない）がすでに保証している。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def _ip(speak):
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip._delivery_block_reason = lambda: ""  # type: ignore[method-assign]
    ip._dif = MagicMock(speak=speak)
    return ip


def _slow_speak():
    """鳴り終わらない声。`release` を立てるまで返らない。"""
    release = asyncio.Event()
    started: list[str] = []

    async def speak(text, *a, **kw):
        started.append(text)
        await release.wait()

    return speak, release, started


# ── 声の終わりを待たずに戻る ──────────────────────────────────────────────


def test_the_filler_returns_before_its_voice_ends():
    async def scenario():
        speak, release, started = _slow_speak()
        ip = _ip(speak)
        await asyncio.wait_for(ip._say_filler("調べてみますね。"), timeout=1.0)
        await asyncio.sleep(0)  # 背景の声を走らせる
        still_speaking = not release.is_set()
        release.set()
        await ip.close()
        return started, still_speaking

    started, still_speaking = asyncio.run(scenario())
    assert started == ["調べてみますね。"], "声は出している"
    assert still_speaking, "鳴り終わる前に戻っている"


def test_the_record_and_screen_do_not_wait_for_the_voice():
    """**画面と記録は声より先。** 以前は声が終わってから吹き出しが出ていた。"""

    async def scenario():
        speak, release, _ = _slow_speak()
        ip = _ip(speak)
        shown: list[str] = []
        ip._emit = lambda t: shown.append(t)  # type: ignore[method-assign]
        await asyncio.wait_for(ip._say_filler("調べてみますね。"), timeout=1.0)
        said = list(ip._req.said_fillers)
        release.set()
        await ip.close()
        return shown, said

    shown, said = asyncio.run(scenario())
    assert shown == ["調べてみますね。"]
    assert said == ["調べてみますね。"]


def test_a_failing_voice_does_not_break_the_turn(caplog):
    """機器は落ちる前提のもの。背景の声が失敗しても、反復は続く。"""

    async def scenario():
        ip = _ip(AsyncMock(side_effect=OSError("スピーカーが無い")))
        with caplog.at_level("WARNING"):
            await asyncio.wait_for(ip._say_filler("調べてみますね。"), timeout=1.0)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        said = list(ip._req.said_fillers)
        await ip.close()
        return said

    assert asyncio.run(scenario()) == ["調べてみますね。"]
    assert any("つなぎの声" in r.getMessage() for r in caplog.records)


def test_close_does_not_wait_for_a_filler_voice():
    """**終わるときは声を待たない。** 鳴り終わらない声でも、後始末は返る。"""

    async def scenario():
        speak, _release, _ = _slow_speak()
        ip = _ip(speak)
        await asyncio.wait_for(ip._say_filler("調べてみますね。"), timeout=1.0)
        await asyncio.sleep(0)
        await asyncio.wait_for(ip.close(), timeout=1.0)

    asyncio.run(scenario())


# ── 調べものがつなぎを待たない ────────────────────────────────────────────


def test_the_lookup_starts_while_the_filler_is_still_speaking():
    """**これが今回の欠陥そのもの。** 実機では検索が 4.59 秒待たされた。"""
    from types import SimpleNamespace

    async def scenario():
        speak, release, _ = _slow_speak()
        ip = _ip(speak)
        started: list[str] = []
        ip._start_lookup = lambda utt, inp, action: started.append(action)  # type: ignore[method-assign]
        decision = SimpleNamespace(
            text="調べてみますね。",
            action="search_deferred",
            query="明日の天気",
            tool_input=None,
        )
        await asyncio.wait_for(
            ip._dispatch_arbiter_action(decision, utterance="明日の天気は"), timeout=1.0
        )
        speaking = not release.is_set()
        release.set()
        await ip.close()
        return started, speaking

    started, speaking = asyncio.run(scenario())
    assert started == ["search_deferred"]
    assert speaking, "検索は、つなぎが鳴っているあいだに投げた"
