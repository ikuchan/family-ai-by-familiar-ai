"""家族のいまの様子は、`FAMILY.md` とは別に DB に持つ（知-ad 段 2・2026-10-01・本人の決定イ）。

性格や好みは変わるが、`FAMILY.md` は人の入力のまま機械は書き換えない（開発ルール・知-ac の季節の層と同じ判断）。
REST が記憶をもとに人ごとに書いた「いまの様子」を `agent_state` の鍵 `family_now` に置き、システム文の
`[家族のいまの様子]` の枠に載せる。人ごとに本文・前の版（1 つ）・数え始めの時刻を持つ。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from familiar_agent.core import family_now as fn
from familiar_agent.core.context_parts import Stance, build_context

T1 = datetime(2026, 9, 1, tzinfo=timezone.utc)
T2 = datetime(2026, 10, 1, tzinfo=timezone.utc)


@pytest.fixture
def clean_state():
    fn.clear()
    yield
    fn.clear()


def test_nothing_is_stored_at_first(clean_state):
    assert fn.stored() == {}
    assert fn.render({}) == ""


def test_a_persons_now_is_kept_with_the_one_before(clean_state):
    assert fn.update("パパ", "サッカーの話が多い。", counted_from=T1)
    assert fn.update("パパ", "最近は釣りに夢中。", counted_from=T2)
    assert fn.update("たいき", "テストが近くて忙しい。", counted_from=T2)
    got = fn.stored()
    assert got["パパ"].text == "最近は釣りに夢中。"
    assert got["パパ"].before == "サッカーの話が多い。"
    assert got["パパ"].counted_from == T2
    assert got["たいき"].before == ""


def test_the_frame_lists_each_person(clean_state):
    fn.update("パパ", "最近は釣りに夢中。", counted_from=T2)
    fn.update("たいき", "テストが近くて忙しい。", counted_from=T2)
    text = fn.render(fn.stored())
    assert text.startswith("[家族のいまの様子]")
    assert "パパ：最近は釣りに夢中。" in text and "たいき：テストが近くて忙しい。" in text
    assert "サッカー" not in text  # 前の版は載せない


def test_the_frame_reaches_the_system_text():
    ctx = build_context(
        stance=Stance.PAJU,
        self_understanding="パジュ",
        family="[家族] パパ",
        family_now="[家族のいまの様子]\nパパ：最近は釣りに夢中。",
    )
    assert "[家族のいまの様子]" in ctx.stable
    assert ctx.stable.index("[一緒に暮らす人たち]") < ctx.stable.index("[家族のいまの様子]")


def test_the_main_llm_and_the_arbiter_and_the_evaluator_get_it(clean_state):
    import asyncio
    from unittest.mock import MagicMock

    from familiar_agent.agent import EmbodiedAgent
    from familiar_agent.backends import ToolCall
    from familiar_agent.core.context_parts import Stance as _S
    from tests._arbiter_fakes import decide, jev_says, system_of, writer_says
    from tests.test_event_loop import _agent, _run, _turn

    fn.update("パパ", "最近は釣りに夢中。", counted_from=T2)

    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    _run(a, utterance="パジュ、こんにちは")
    assert "[家族のいまの様子]" in "\n".join(a.backend.stream_turn.call_args.kwargs["system"])

    writer = writer_says({"text": "はい"})
    asyncio.run(
        decide(
            jev=jev_says("light"),
            writer=writer,
            utterance="こんにちは",
            family_now=fn.render(fn.stored()),
        )
    )
    assert "[家族のいまの様子]" in system_of(writer)

    e = MagicMock(spec=EmbodiedAgent)
    e._me_md, e._family_md, e._people_md = "パジュ", "[家族] パパ", ""
    assert "[家族のいまの様子]" in (EmbodiedAgent._stance_context(e, _S.PAJU) or "")
