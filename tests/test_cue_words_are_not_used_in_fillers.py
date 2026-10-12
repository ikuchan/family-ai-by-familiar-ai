"""合図で言った言葉は、つなぎに使わない言葉として書き手に渡す（出-ca・2026-10-13・本人の決定）。

出-bt ③ で、合図の作り置きの声（はい・うん など）を「言ったつなぎ」に数えた。すると「どちらかがもう片方を含めば同じ
つなぎ」の判定に引っかかり、10/13 00:35 には「はい、少々お待ちくださいね。」が捨てられ、答えまで約 15 秒声が無かった。
同じつなぎの判定は変えず、合図の言葉は「つなぎに使わない言葉」として書き手に渡す（本人）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from familiar_agent.loop.arbiter import ArbiterInput
from familiar_agent.loop.event_loop import InformationProcessing
from tests._arbiter_fakes import prompt_of, writer_says
from tests.test_event_loop import _agent


def _write(words):
    from familiar_agent.loop.arbiter import Arbiter

    writer = writer_says({"filler": "少々お待ちくださいね"})
    arbiter = Arbiter(jev=None, writer=writer)
    asyncio.run(
        arbiter.write_filler(
            ArbiterInput(utterance="明日の天気は？", workspace_ctx=""),
            "いまは返事を考えている最中",
            avoid=words,
        )
    )
    return prompt_of(writer)


def test_the_writer_is_told_not_to_use_the_cue_word():
    assert "合図でもう言ったので、つなぎに使わない：「はい」" in _write(["はい"])


def test_no_cue_word_no_line():
    """**反証の側**：合図が効果音（言葉なし）なら、その行は載せない。"""
    assert "つなぎに使わない" not in _write([])


def test_a_filler_with_the_cue_word_is_not_dropped_as_a_repeat():
    """同じつなぎの判定は、合図の言葉を知らない。「はい、…」のつなぎも捨てない。"""
    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._req.cue_words.append("はい")
    ip._delivery_block_reason = lambda: ""  # type: ignore[method-assign]
    ip._window_closed_for_conversation = lambda: False  # type: ignore[method-assign]
    ip._speak_filler_in_background = MagicMock()  # type: ignore[method-assign]

    async def go():
        await ip._say_filler("はい、少々お待ちくださいね。")
        await ip.close()

    asyncio.run(go())
    assert ip._req.said_fillers == ["はい、少々お待ちくださいね。"]


def test_the_background_filler_passes_the_cue_words(monkeypatch):
    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._req.cue_words.append("うん")
    ip._waiting_on = lambda: [MagicMock(action="主LLM")]  # type: ignore[method-assign]
    got: list = []

    async def write(inp, what, avoid=()):
        got.append(list(avoid))
        return ""

    ip._arbiter = lambda: MagicMock(write_filler=write)  # type: ignore[method-assign]
    ip._arbiter_input = lambda **k: None  # type: ignore[method-assign]

    async def go():
        await ip._say_waiting_filler("x", "", 1)
        await asyncio.gather(*ip._filler_jobs)
        await ip.close()

    asyncio.run(go())
    assert got == [["うん"]]
