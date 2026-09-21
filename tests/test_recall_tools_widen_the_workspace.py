"""思い出し方を変える 4 つの道具（出-ah・2026-09-21）。

W に載る情報が足りないとき、主LLM が**面・件数と思い出し方・時期と幅・直近の窓**を変えて
引き直せるようにする。効き目はその 1 回だけで、以後の反復の W は元の条件で組む。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from familiar_agent.loop.event_loop import _FULL_ACTIONS, InformationProcessing
from tests.test_event_loop import _agent as _base_agent, _turn

TOOLS = ("recall_as", "recall_deeper", "recall_when", "recall_recent")


def _agent():
    return _base_agent(stream_returns=[_turn([])])


def test_the_four_tools_are_offered_to_the_main_llm():
    for name in TOOLS:
        assert name in _FULL_ACTIONS, name
    ip = InformationProcessing(_agent())
    names = {d.get("name") for d in ip._tools(actions=_FULL_ACTIONS, cache_tools=False)}
    assert set(TOOLS) <= names


def test_each_tool_says_when_to_use_it():
    ip = InformationProcessing(_agent())
    defs = {d["name"]: d for d in ip._tools(actions=_FULL_ACTIONS, cache_tools=False)}
    for name in TOOLS:
        text = defs[name].get("description", "")
        assert "思い出せない" in text, f"{name} に使いどころが書かれていない"


@pytest.mark.parametrize(
    "action,tool_input,expect",
    [
        ("recall_as", {"person": "パパ", "query": "海"}, "パパの面"),
        ("recall_deeper", {"query": "海", "way": "新しい順に", "n": 50}, "20 件"),
        ("recall_when", {"query": "海", "date": "2026-08-15", "span_days": 45}, "2026-08-15"),
        ("recall_recent", {"minutes": 90, "turns": 50}, "30 分"),
    ],
)
def test_a_tool_returns_what_it_widened(action, tool_input, expect):
    """返りは `recall` と同じ形＋母数。面を変えた道具は誰の面かを必ず言う。"""
    ip = InformationProcessing(_agent())
    ip._agent._pmm.find_person_id_by_name = MagicMock(return_value="pid-papa")
    ip._agent._pmm.get_person_name = MagicMock(return_value="パパ")
    ws = MagicMock()
    ws.memories = []
    ws.render = MagicMock(
        return_value="[過去の記憶（証拠つき）: conf<0.55 は不確か]:\n- 去年の夏の話"
    )
    ws.recent_text = MagicMock(return_value="[直近のやりとり]\n- さっきの話")
    with patch("familiar_agent.loop.workspace.recall", new=AsyncMock(return_value=ws)):
        out = _run(ip, action, tool_input)
    assert expect in out


def _run(ip, action: str, tool_input: dict) -> str:
    import asyncio

    async def go():
        await ip._run_lookup_body(action, tool_input, "しらべ", None)
        return ip._triggers.get_nowait().result

    return asyncio.run(go())


def test_the_widening_does_not_leak_into_the_next_iteration():
    """効き目はその 1 回だけ（`Request` に何も残さない）。"""
    ip = InformationProcessing(_agent())
    ip._agent._pmm.find_person_id_by_name = MagicMock(return_value="pid-papa")
    ws = MagicMock()
    ws.memories = []
    ws.render = MagicMock(return_value="[過去の記憶（証拠つき）: conf<0.55 は不確か]:")
    ws.recent_text = MagicMock(return_value="")
    before = dict(vars(ip._req))
    with patch("familiar_agent.loop.workspace.recall", new=AsyncMock(return_value=ws)):
        _run(ip, "recall_deeper", {"query": "海", "n": 20})
    after = dict(vars(ip._req))
    changed = {
        k for k in after if k not in ("lookups", "turn_records") and after[k] != before.get(k)
    }
    assert not changed, f"求めに残ってしまった：{changed}"


# ── 足りないことに気づく手がかり（出-ah 段 3）──────────────────────────────


def test_the_workspace_says_how_it_recalled():
    """W の記憶の列の先頭に、いまの引き方を 1 行。足りないと気づく手がかりになる。"""
    from familiar_agent.loop import workspace
    from familiar_agent.loop.request import Request

    oif = MagicMock()
    oif.actors = MagicMock(return_value={})
    oif.roles = MagicMock(return_value={})
    from tests.test_workspace_is_the_core import _rec

    text, _ = workspace.compose(
        oif, [_rec("m1", "去年の夏の話")], Request(), basis="いまの相手の面・7 件・いま基準"
    )
    assert "[この想起：いまの相手の面・7 件・いま基準]" in text
    assert text.index("[この想起") < text.index("[過去の記憶")


def test_without_a_basis_nothing_is_added():
    from familiar_agent.loop import workspace
    from familiar_agent.loop.request import Request

    oif = MagicMock()
    oif.actors = MagicMock(return_value={})
    oif.roles = MagicMock(return_value={})
    from tests.test_workspace_is_the_core import _rec

    text, _ = workspace.compose(oif, [_rec("m1", "去年の夏の話")], Request())
    assert "[この想起" not in text


def test_the_rules_allow_widening_before_giving_up():
    from familiar_agent.loop.prompt import CHECKER_RULE_IDS, rules_section

    rules = rules_section()
    assert "widen-before-giving-up" in rules
    assert "諦める前に" in rules
    # 文だけでは違反を判定できないので、整合チェックへは渡さない
    assert "widen-before-giving-up" not in CHECKER_RULE_IDS
