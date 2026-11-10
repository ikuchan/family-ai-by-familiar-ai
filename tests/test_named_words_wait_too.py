"""求めが開いているあいだは、名前があってもなくても割り込まない（環-aw 決定ホ・第 1 段 項目 9・2026-11-10 本人の決定 B）。

門を通る名前つきの言葉（キーボード、鳴っている最中の声）も、名前の無い言葉と同じに扱う。主LLM が考えている最中は
待たせ、返りの時点で考え直すかを決める。調べものだけが動いているなら、いまの求めに添える。新しい求めを開いて
前の求めの上に重ねない（重ねると、前の主LLM が動き続け、待ちの電子音が二重になり、前の求めが閉じない）。
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, patch

from familiar_agent.loop import event_loop as el
from familiar_agent.loop.event_loop import InformationProcessing, Trigger
from familiar_agent.loop.request import Lookup

from tests.test_event_loop import _agent


def _ip(action: str):
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip._req.request_id = "req-1"
    ip._req.trigger_kind = "発話"
    ip._req.lookups = [Lookup(index=1, action=action, query="q", generation=0)]
    return ip, a


def _named(text: str, source: str = "keyboard") -> Trigger:
    fut = asyncio.get_running_loop().create_future()
    return Trigger(kind="会話入力", query=text, source=source, named=True, future=fut)


def test_a_named_keyboard_word_waits_while_the_main_llm_thinks():
    ip, _ = _ip("主LLM")

    async def go():
        ip._triggers.put_nowait(_named("パジュ、やっぱり明後日"))
        ip._triggers.put_nowait(Trigger(kind="完了", query="主LLM1", result="{}"))
        return await asyncio.wait_for(ip._take_trigger(), timeout=1.0)

    assert asyncio.run(go()) is None  # 先に主LLM の返りを取り込む。言葉は待たせる
    assert [t.query for t in ip._held] == ["パジュ、やっぱり明後日"]


def test_a_named_word_is_asked_whether_to_rethink():
    """待っている名前つきの言葉も、返りの時点で考え直すかに掛ける（`_waiting_words`）。"""
    ip, _ = _ip("主LLM")

    async def go():
        ip._held.append(_named("パジュ、やっぱり明後日"))
        return ip._waiting_words()

    assert [t.query for t in asyncio.run(go())] == ["パジュ、やっぱり明後日"]


def test_a_named_word_during_a_lookup_joins_the_request():
    """調べものだけが動いているなら、名前つきでも新しい求めを開かずに添える。"""
    ip, _ = _ip("search_deferred")
    ip._attach_to_request = AsyncMock()
    ip._begin_utterance = AsyncMock()

    async def go():
        trig = _named("パジュ、東京のね")
        ip._triggers.put_nowait(trig)
        driver = asyncio.ensure_future(ip._drive())
        for _ in range(50):
            await asyncio.sleep(0.01)
            if ip._attach_to_request.await_count or ip._begin_utterance.await_count:
                break
        driver.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await driver
        return trig

    trig = asyncio.run(go())
    ip._attach_to_request.assert_awaited_once_with(trig)
    ip._begin_utterance.assert_not_awaited()
    assert ip._req.request_id == "req-1"


def test_the_time_out_goes_before_a_named_word():
    """時間切れの知らせは、名前つきの言葉より先に返す。言葉は待たせ、答えのあとで新しい求めになる。"""
    ip, _ = _ip("主LLM")
    t = Trigger(kind=el.TIMED_OUT_KIND, query="req-1")

    async def go():
        words = _named("パジュ、止めて", source="voice")
        ip._triggers.put_nowait(t)
        ip._triggers.put_nowait(words)
        return await ip._take_trigger(), words

    first, words = asyncio.run(go())
    assert first is t
    assert words in ip._held


def test_an_attached_named_word_marks_the_request_named():
    """名前つきの言葉を添えたら、その求めは名前で呼ばれたことにする（出-db 段 B の読み直しと、話し終わりの窓）。"""
    ip, _ = _ip("search_deferred")
    ip._req.named = False

    async def go():
        await ip._attach_to_request(_named("パジュ、それで？"))

    asyncio.run(go())
    assert ip._req.named is True


def test_an_attached_unnamed_word_leaves_the_request_as_it_was():
    """**反証の側**：名前の無い言葉では変えない。"""
    ip, _ = _ip("search_deferred")
    ip._req.named = False

    async def go():
        fut = asyncio.get_running_loop().create_future()
        await ip._attach_to_request(
            Trigger(kind="会話入力", query="それで？", named=False, future=fut)
        )

    asyncio.run(go())
    assert ip._req.named is False


# ── 人の言葉から始まっていない求め（情動・機器）には添えない（2026-11-10 本人の決定ア） ─────────────
# 情動の求めは、人と話していなければ声にしない（独り言）。機器の求めは「人の言葉ではない。直近のやりとりは済んだこと」
# と W に書く。どちらに添えても、添えた言葉に答えが返らないおそれがある。門を通った言葉は、名前があってもなくても
# 待たせ、その求めが閉じてから会話の求めにする。


def _not_spoken_to(kind: str, action: str = "search_deferred"):
    ip, a = _ip(action)
    ip._req.trigger_kind = kind
    return ip, a


def _take_with_a_completion(ip, word: Trigger):
    async def go():
        ip._triggers.put_nowait(word)
        ip._triggers.put_nowait(Trigger(kind="完了", query="q", result="{}"))
        return await asyncio.wait_for(ip._take_trigger(), timeout=1.0)

    return asyncio.run(go())


def test_a_word_during_a_device_lookup_waits():
    ip, _ = _not_spoken_to("機器")

    async def make():
        return _named("パジュ、明日の天気は？", source="voice")

    word = asyncio.run(make())
    assert _take_with_a_completion(ip, word) is None  # 先に調べものの返りを取り込む
    assert [t.query for t in ip._held] == ["パジュ、明日の天気は？"]


def test_an_unnamed_word_during_an_affect_lookup_waits():
    """門を通った名前の無い言葉（窓の中・音楽の言葉）も同じ。"""
    ip, _ = _not_spoken_to("情動")
    word = Trigger(kind="会話入力", query="ストップ", source="voice", named=False)
    assert _take_with_a_completion(ip, word) is None
    assert [t.query for t in ip._held] == ["ストップ"]


def test_a_word_during_a_conversation_lookup_is_still_taken():
    """**反証の側**：会話の求めのあいだは、いままでどおり取り出して添える。"""
    ip, _ = _not_spoken_to("発話")
    word = Trigger(kind="会話入力", query="ストップ", source="voice", named=False)
    assert _take_with_a_completion(ip, word) is word
    assert ip._held == []


def test_an_affect_request_does_not_rethink_with_waiting_words():
    """主LLM が返っても、情動・機器の求めには考え直して添えない。言葉は答えのあとで新しい求めになる。"""
    from familiar_agent.core import jev_judges

    for kind in ("情動", "機器"):
        ip, _ = _not_spoken_to(kind, "主LLM")
        ip._held.append(Trigger(kind="会話入力", query="うん", named=False))
        judge = AsyncMock(return_value=jev_judges.RETHINK)
        with patch.object(jev_judges, "judge_rethink", judge):
            got = asyncio.run(ip._rethink_or_speak("下書き"))
        assert got == jev_judges.AS_IS, kind
        judge.assert_not_awaited()


def test_a_conversation_request_still_asks_whether_to_rethink():
    """**反証の側**：会話の求めでは、いままでどおり Jev に聞く。"""
    from familiar_agent.core import jev_judges

    ip, _ = _not_spoken_to("発話", "主LLM")
    ip._held.append(Trigger(kind="会話入力", query="うん", named=False))
    judge = AsyncMock(return_value=jev_judges.RETHINK)
    with patch.object(jev_judges, "judge_rethink", judge):
        got = asyncio.run(ip._rethink_or_speak("下書き"))
    assert got == jev_judges.RETHINK
    judge.assert_awaited_once()


def test_the_waiting_word_comes_out_once_the_device_request_is_done():
    """待たせた言葉は、その求めの調べものが終われば取り出す（待たせた印つき）。駆動体はそれを新しい会話の求めにする。"""
    ip, _ = _not_spoken_to("機器")
    word = Trigger(kind="会話入力", query="ストップ", source="voice", named=False)
    assert _take_with_a_completion(ip, word) is None
    ip._req.lookups = []  # 返りを取り込み、飛んでいるものが無くなった

    async def go():
        return await asyncio.wait_for(ip._take_trigger(), timeout=1.0)

    got = asyncio.run(go())
    assert got is word and got.waited is True
    assert ip._held == []


def test_a_confirmation_reply_is_still_taken_during_a_device_lookup():
    """確認待ち（知-o）の返事は、機器の求めのあいだでも取り出す。駆動体が 3 択の求めにする（いままでどおり）。"""
    ip, a = _not_spoken_to("機器")
    a.confirm_alive = lambda: True
    word = Trigger(kind="会話入力", query="はい", source="voice", named=False)
    assert _take_with_a_completion(ip, word) is word
