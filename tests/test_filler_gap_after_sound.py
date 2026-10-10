"""2 つ目以降のつなぎは、最後に音が鳴り終わってから 3 秒黙ったら出す（出-bd ②・2026-10-10・本人の決定「三秒」）。

出-bc で、最初のつなぎは人の言葉から通しで 5 秒に直した。それでも 2 つ目は前のつなぎから 20 秒（以前の設定）
だったので、10/08 21:23「明日の天気は？」ではつなぎが鳴り終わってから答えまで 8.3 秒黙った。合図の声（出-bg の「はい」など）を
流した求めでは最初のつなぎも省いていたので、「はい」のあと答えまで 12 秒ほど黙る計算になる。

見張りが次を出す時刻は、次の遅いほう：話しかけられてから 5 秒・前のつなぎを決めてから 3 秒・最後に音が鳴り終わってから 3 秒。
つなぎや合図がまだ鳴っているあいだは出さない。合図の声も音に数え、合図のあとの最初のつなぎは省かない。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.event_loop import InformationProcessing, Lookup

from tests.test_event_loop import _agent


def _ip(*, first=0.05, gap=0.2):
    a = _agent(stream_returns=[])
    a.config.lookup_slow_seconds = first
    a.config.wait_filler_gap_seconds = gap
    ip = InformationProcessing(a)
    ip._req.trigger_kind = "発話"
    ip._req.request_id = "req-1"
    ip._req.lookups = [Lookup(index=1, action="search_deferred", query="q", generation=0)]
    return ip


def _drain(ip) -> int:
    n = 0
    while not ip._triggers.empty():
        n += ip._triggers.get_nowait().kind == "進捗"
    return n


def _counts(ip, *, setup, checks):
    """見張りを立て、`checks` の秒ごとに積まれた「進捗」の数（累計）を返す。"""

    async def go():
        setup()
        ip._ensure_wait_watch()
        out, seen, t0 = [], 0, time.monotonic()
        for at in checks:
            await asyncio.sleep(max(0.0, t0 + at - time.monotonic()))
            seen += _drain(ip)
            out.append(seen)
        ip._stop_wait_watch()
        for task in list(ip._filler_voices):
            task.cancel()  # 鳴りっぱなしの偽のつなぎを片付ける
        await ip.close()
        return out

    return asyncio.run(go())


def test_the_next_filler_comes_a_gap_after_the_last_sound():
    ip = _ip()

    def setup():
        now = time.monotonic()
        ip._req.heard_at = now - 1.0
        ip._req.last_progress_at = now - 0.5  # 前のつなぎを決めた
        ip._req.last_sound_ended_at = now  # そのつなぎがいま鳴り終わった

    assert _counts(ip, setup=setup, checks=(0.1, 0.3)) == [0, 1]


def test_nothing_while_a_filler_is_still_sounding():
    ip = _ip()

    def setup():
        now = time.monotonic()
        ip._req.heard_at = now - 1.0
        ip._req.last_progress_at = now - 1.0
        ip._req.last_sound_ended_at = now - 1.0
        ip._filler_voices.add(asyncio.ensure_future(asyncio.sleep(10)))  # まだ鳴っている

    assert _counts(ip, setup=setup, checks=(0.3,)) == [0]


def test_the_cue_voice_counts_as_a_sound():
    ip = _ip()

    def setup():
        now = time.monotonic()
        ip._req.heard_at = now - 1.0  # 最初の 5 秒（ここでは 0.05）はもう過ぎた
        ip._req.last_sound_ended_at = now  # 合図の「はい」がいま鳴り終わった

    assert _counts(ip, setup=setup, checks=(0.1, 0.3)) == [0, 1]


def test_the_cue_writes_when_it_ended():
    from familiar_agent.core import reaction_cue as rc
    from familiar_agent.loop.event_loop import _ReactionCue

    ip = InformationProcessing(_agent(stream_returns=[]))

    async def go():
        done = asyncio.get_running_loop().create_future()
        ip._dif.play_cue = lambda cue: done  # type: ignore[method-assign]
        _ReactionCue(ip, MagicMock()).decided("play_music")  # TOOL：作り置きの声
        assert rc.cue_for("play_music") == rc.TOOL
        assert ip._req.last_sound_ended_at is None
        before = time.monotonic()
        done.set_result(None)
        await asyncio.sleep(0)
        await ip.close()
        return before

    before = asyncio.run(go())
    assert ip._req.last_sound_ended_at is not None and ip._req.last_sound_ended_at >= before


def test_the_first_filler_is_not_skipped_after_the_cue_voice():
    ip = InformationProcessing(_agent(stream_returns=[]))
    said: list[str] = []
    ip._waiting_on = lambda: [MagicMock(action="主LLM")]  # type: ignore[method-assign]

    async def say(text):
        said.append(text)

    ip._say_filler = say  # type: ignore[method-assign]
    ip._arbiter = lambda: MagicMock(write_filler=AsyncMock(return_value="もう少しね"))  # type: ignore[method-assign]
    ip._arbiter_input = lambda **k: None  # type: ignore[method-assign]  # 材料は見ない（本物は組むのに 14 秒かかる）

    async def go():
        ip._req.last_sound_ended_at = time.monotonic()  # 合図の声が鳴った
        await ip._say_waiting_filler("x", "", 1)
        await ip.close()

    asyncio.run(go())
    assert said == ["もう少しね"]


def test_the_gap_comes_from_config():
    import os
    from unittest.mock import patch

    from familiar_agent.config import AgentConfig

    with patch.dict(os.environ, {}, clear=True):
        assert AgentConfig().wait_filler_gap_seconds == 3.0
