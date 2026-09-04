"""家の決まりを引く道具を主LLM へ渡す（`設計方針_家の記録との接続` §5）。

**繋ぐのは家族ティアの1本だけである。** `get_house_rules` は話者ゲートが要らない
（家族の誰と話していても引ける）。`ask_vault_yusuke` は Vault を**全部**読むので、
**話者ゲートの仕組みができるまで道具一覧に出さない**——存在しない道具は呼べない、が
唯一の確実な守り方である。

**サーバー側からは誰が話しているか見えない。** だからゲートはこちらが掛ける。規則は
名前で決まる：**名前に人が入っている道具（`*_yusuke`）は、その人のターン以外では出さない。**
将来 `*_taeko` が増えても同じ規則で通る。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from familiar_agent.loop.event_loop import InformationProcessing


def _agent_with_mcp(names: list[str]):
    a = MagicMock()
    a._mcp = MagicMock()
    a._mcp.get_tool_definitions.return_value = [
        {"name": n, "description": f"{n} の説明", "input_schema": {"type": "object"}}
        for n in names
    ]
    return a


def _tool_names(agent, actions: tuple[str, ...]) -> set[str]:
    ip = InformationProcessing.__new__(InformationProcessing)
    ip._agent = agent
    return {d["name"] for d in ip._tools(actions=actions)}


# ── ① 家族ティアは渡る ─────────────────────────────────────────────────────

def test_the_house_rules_tool_is_offered() -> None:
    got = _tool_names(_agent_with_mcp(["get_house_rules"]), ("house_rules",))
    assert got == {"get_house_rules"}


# ── ② 名前に人が入っている道具は渡さない ───────────────────────────────────

def test_a_personal_tool_is_never_offered_yet() -> None:
    """**話者ゲートができるまで、個人ティアは出さない。**

    `ask_vault_yusuke` は転職の悩み・人物評・家計・子どもについて父が書いたことを読める。
    子どもと話しているターンで存在してはいけない。
    """
    agent = _agent_with_mcp(["get_house_rules", "ask_vault_yusuke"])
    got = _tool_names(agent, ("house_rules",))
    assert "ask_vault_yusuke" not in got, "個人ティアの道具が主LLM へ渡っている"
    assert got == {"get_house_rules"}


def test_the_rule_is_by_name_not_by_a_fixed_list() -> None:
    """将来 `*_taeko` が増えても同じ規則で落ちる（名前に人が入っているか）。"""
    agent = _agent_with_mcp(["get_house_rules", "ask_vault_taeko", "ask_vault_yusuke"])
    assert _tool_names(agent, ("house_rules",)) == {"get_house_rules"}


# ── ③ MCP が無い構成でも壊れない ───────────────────────────────────────────

def test_no_mcp_means_no_tool() -> None:
    a = MagicMock()
    a._mcp = None
    assert _tool_names(a, ("house_rules",)) == set()


# ── ④ 主LLM へ実際に渡る（動作の一覧に載っている）─────────────────────────

def test_the_house_rules_action_reaches_the_full_llm() -> None:
    """`_ACTIONS` に置いただけでは渡らない。連鎖が続く反復の一覧にも要る。"""
    from familiar_agent.loop.event_loop import _FULL_ACTIONS

    assert "house_rules" in _FULL_ACTIONS


def test_asking_the_rules_ends_the_iteration() -> None:
    """**1反復1出力を守る。** 即座に返るが、それを見て何を言うかは次の反復が決める
    （`recall`・`see` と同じ扱い）。ここに入れないと、同じ反復で発話まで進んでしまう。
    """
    from familiar_agent.loop.event_loop import _LOOKUP_ACTIONS

    assert "house_rules" in _LOOKUP_ACTIONS
