"""聞かれたとき Vault の中身を引く（知-g-い・`設計方針_家の記録との接続` §2・2026-09-15）。

`ask_vault_<英字>`（個人ティア・数十秒）を、動作 `vault` として主LLM と調停の候補に載せる。
道具の名前は話者の英字から引き、話者ゲート（知-f）が本人以外を落とす。遅さは既存の
投げっぱなし（背景タスク・完了は待ち行列）で足りるが、MCP の時間切れだけは長くする。
返りの末尾の `──（YYYY-MM-DD のセッション／継続）` は素性の印なので落とす。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.io.dif import DIF, strip_session_mark
from familiar_agent.loop.arbiter import _EXTRA_ACTIONS, _parse
from familiar_agent.loop.event_loop import (
    _MCP_LOOKUPS,
    InformationProcessing,
    _action_family,
    _query_label,
    _tool_input_for,
)
from familiar_agent.mcp_client import call_timeout_for

FAMILY = "## ゆうすけ\n- **名前**：ゆうすけ\n- **呼び方**：パパ\n- **英字**：Yusuke Ikunaga\n\n## たいき\n- **名前**：たいき\n- **呼び方**：たいき\n- **英字**：Taiki Ikunaga\n"


def _defs(*names: str) -> list[dict]:
    return [{"name": n, "description": "…", "input_schema": {"type": "object"}} for n in names]


def test_the_tool_name_is_a_family_of_the_vault_action():
    assert _action_family("ask_vault_yusuke") == "vault"
    assert _action_family("ask_vault_taeko") == "vault"
    assert "vault" in _MCP_LOOKUPS


def test_the_label_carries_the_question_and_the_input_is_the_question():
    assert (
        _query_label("vault", {"question": "コーチングどうなった？"})
        == "記録に「コーチングどうなった？」を聞く"
    )
    assert _query_label("ask_vault_yusuke", {"question": "x"}) == "記録に「x」を聞く"
    assert _tool_input_for("vault", "コーチングどうなった？") == {
        "question": "コーチングどうなった？"
    }


def _ip(speaker: str, tools: list[str]):
    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    a._family_md = FAMILY
    ip = InformationProcessing(a)
    ip._current_speaker_name = lambda: speaker  # type: ignore[method-assign]
    ip._dif = MagicMock()
    ip._dif.tool_defs_with_prefix = lambda prefix: [
        d for d in _defs(*tools) if d["name"].startswith(prefix)
    ]
    ip._dif.tool_defs = lambda name: [d for d in _defs(*tools) if d["name"] == name]
    return ip


def test_the_vault_tool_is_offered_only_to_its_owner():
    ip = _ip("ゆうすけ", ["get_house_rules", "ask_vault_yusuke"])
    assert [d["name"] for d in ip._tools(actions=("vault",))] == ["ask_vault_yusuke"]
    assert "vault" in ip._extra_actions()
    ip = _ip("たいき", ["get_house_rules", "ask_vault_yusuke"])
    assert ip._tools(actions=("vault",)) == []
    assert "vault" not in ip._extra_actions()
    ip = _ip("", ["ask_vault_yusuke"])
    assert "vault" not in ip._extra_actions()


def test_the_lookup_resolves_the_tool_from_the_speaker():
    """`vault` は話者の英字で `ask_vault_<latin>` に解ける。誰か分からなければ呼べない。"""
    ip = _ip("ゆうすけ", ["ask_vault_yusuke"])
    assert ip._vault_tool_name() == "ask_vault_yusuke"
    ip = _ip("たいき", ["ask_vault_yusuke"])
    assert ip._vault_tool_name() == ""  # たいきの道具は繋がっていない


def test_the_arbiter_offers_the_vault_with_a_question():
    d = _parse(
        '{"branch": "action", "action": "vault", "query": "コーチングの返事どうなった？", "text": "見てくるね"}',
        can_see=False,
        extra_actions=("vault",),
    )
    assert d is not None and d.action == "vault" and d.query == "コーチングの返事どうなった？"
    assert "vault" in _EXTRA_ACTIONS and _EXTRA_ACTIONS["vault"][0] == ""  # query が要る


def test_the_session_mark_at_the_tail_is_dropped():
    text = "コーチの返事は来ていない。\n──（2026-09-15 のセッション／継続）"
    assert strip_session_mark(text) == "コーチの返事は来ていない。"
    assert strip_session_mark("そのまま") == "そのまま"


def test_call_tool_strips_the_mark_and_the_slow_tool_gets_a_longer_timeout():
    mcp = MagicMock()
    mcp.call_result = AsyncMock(
        return_value=MagicMock(
            text="答え\n──（2026-09-15 のセッション／新規）", image=None, ok=True
        )
    )
    text, ok = asyncio.run(DIF(mcp=mcp).call_tool("ask_vault_yusuke", {"question": "q"}))
    assert ok and text == "答え"
    assert call_timeout_for("ask_vault_yusuke") >= 120.0
    assert call_timeout_for("get_house_rules") == 30.0


def test_notion_and_vault_are_told_apart_in_the_candidates():
    """書き分け：Notion＝家の目次・日次記録・Todo、Vault＝本人の考え・経緯・検討の中身（2026-09-15）。

    両方に「記録・経緯」と書いてあると Notion（速い）に寄る（実機で 2 回とも Notion へ行った）。
    """
    notion = _EXTRA_ACTIONS["notion_search"][1]
    vault = _EXTRA_ACTIONS["vault"][1]
    assert "目次" in notion and "Todo" in notion and "経緯" not in notion
    assert "経緯" in vault and "検討" in vault and "目次や Todo ではない" in vault
