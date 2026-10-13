"""つなぎを続けて頼まない・答えの後に言わない・合図の言葉を覚える（出-bt・2026-10-12・本人の決定）。

10/12 22:15「明日の天気は?」：見張りはつなぎ 1 を頼んだ時刻（53.26）から 3 秒で、まだ何も鳴っていないので
つなぎ 2 を頼んだ（56.27）。頼んでから声になるまで（想起・書き手・合成で 3.5 秒）が、つなぎの間（3 秒）より長かった。
つなぎ 2 の文ができたのは答えが届いた後（58.49 → 58.67）で、見直さずに言い、答えは約 10 秒遅れた。
合図で「はい」と言ったことも書き手は知らず、つなぎは 2 つとも「はい、」で始まった。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.event_loop import InformationProcessing, Lookup, Trigger

from tests.test_event_loop import _agent
from tests.test_filler_gap_after_sound import _counts, _ip

# ── ① 書いているつなぎがあれば、次を頼まない ─────────────────────────────


def test_no_next_filler_while_one_is_being_written():
    ip = _ip()

    def setup():
        now = time.monotonic()
        ip._req.heard_at = now - 1.0
        ip._req.last_progress_at = now - 1.0  # 前のつなぎを頼んでから、間（0.2 秒）はもう過ぎた
        ip._req.filler_pending = True  # まだ書いている（声にも取り下げにもなっていない）

    assert _counts(ip, setup=setup, checks=(0.3,)) == [0]


def test_the_watch_marks_the_filler_it_asked_for():
    """**反証の側**：頼んでいるつなぎが無ければ頼み、頼んだら印を立てる。"""
    ip = _ip()

    def setup():
        now = time.monotonic()
        ip._req.heard_at = now - 1.0
        ip._req.last_progress_at = now - 1.0

    assert _counts(ip, setup=setup, checks=(0.3,)) == [1]
    assert ip._req.filler_pending is True


def test_the_mark_is_lowered_when_a_result_comes_with_it():
    """つなぎの頼みと本物の結果が一緒に届くと、つなぎの反復は起きない。印は取込で下ろす。"""
    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._req.filler_pending = True
    ip._req.lookups = [Lookup(index=1, action="search_deferred", query="q", generation=0)]
    ip._drained_completions = [
        Trigger(kind="進捗", query="q"),
        Trigger(kind="完了", query="q", result="晴れ", index=1),
    ]

    async def go():
        with_oif = ip._agent._oif
        with_oif.write = AsyncMock(return_value="obs-1")
        await ip._intake()
        await ip.close()

    asyncio.run(go())
    assert ip._req.filler_pending is False


# ── ② 書いているあいだに答えが届いたら、言わない ─────────────────────────


def _filler_ip(*, still_waiting_after_write: bool):
    ip = InformationProcessing(_agent(stream_returns=[]))
    said: list[str] = []
    waiting = [MagicMock(action="主LLM")]
    ip._waiting_on = lambda: waiting  # type: ignore[method-assign]

    async def write(inp, what, avoid=()):
        if not still_waiting_after_write:
            waiting.clear()  # 書いているあいだに答えが届いた
        return "もう少しね"

    async def say(text):
        said.append(text)

    ip._say_filler = say  # type: ignore[method-assign]
    ip._arbiter = lambda: MagicMock(write_filler=write)  # type: ignore[method-assign]
    ip._arbiter_input = lambda **k: None  # type: ignore[method-assign]
    ip._req.filler_pending = True
    return ip, said


def test_a_filler_written_after_the_answer_is_not_said(caplog):
    ip, said = _filler_ip(still_waiting_after_write=False)

    async def go():
        await ip._say_waiting_filler("x", "", 1)
        await asyncio.gather(*ip._filler_jobs)  # 書き手と声は裏の仕事（出-bx）
        await ip.close()

    with caplog.at_level("INFO"):
        asyncio.run(go())
    assert said == []
    assert "答えが届いたのでつなぎを言わない" in caplog.text
    assert ip._req.filler_pending is False


def test_a_filler_is_said_while_still_waiting():
    """**反証の側**：まだ待たせていれば言い、印を下ろす。"""
    ip, said = _filler_ip(still_waiting_after_write=True)

    async def go():
        await ip._say_waiting_filler("x", "", 1)
        await asyncio.gather(*ip._filler_jobs)  # 書き手と声は裏の仕事（出-bx）
        await ip.close()

    asyncio.run(go())
    assert said == ["もう少しね"]
    assert ip._req.filler_pending is False


# ── ③ 合図で言った作り置きの言葉を覚える ─────────────────────────────


def _decide(final: str, monkeypatch) -> "list[str]":
    from familiar_agent.io import dif as dif_mod
    from familiar_agent.loop.event_loop import _ReactionCue

    async def _quiet(path, gain):
        return True

    monkeypatch.setattr(dif_mod, "_play_wav", _quiet)
    ip = InformationProcessing(_agent(stream_returns=[]))

    async def go():
        _ReactionCue(ip, MagicMock()).decided(final)
        await asyncio.sleep(0)
        await ip.close()

    asyncio.run(go())
    assert (
        ip._req.said_fillers == []
    )  # 言ったつなぎには数えない（出-ca・同じつなぎの判定に引っかかった）
    return list(ip._req.cue_words)


def test_the_cue_word_is_kept_as_a_word_not_to_use(monkeypatch):
    assert _decide("play_music", monkeypatch) in (["はい"], ["うん"])


def test_an_effect_sound_is_not_a_word(monkeypatch):
    """**反証の側**：効果音 B には言葉が無いので、何も覚えない。"""
    assert _decide("reply_light", monkeypatch) == []


def test_word_for_maps_the_files():
    from familiar_agent.core import reaction_cue as rc

    assert rc.word_for(rc.TOOL_WAVS[0]) == "はい"
    assert rc.word_for(rc.THINK_WAVS[0]) == "えっと"  # 「んー」を外して詰めた（出-ce）
    assert rc.word_for(rc.ACK_WAV) == ""


# ── ④ 書き手は、渡された様子だけを伝える ─────────────────────────────


def test_the_writer_follows_the_given_state():
    from familiar_agent.loop.arbiter import WRITER_PROMPT

    assert "いま考えている・調べている最中だと伝えるもの" not in WRITER_PROMPT
    assert "渡された様子だけを伝える" in WRITER_PROMPT
