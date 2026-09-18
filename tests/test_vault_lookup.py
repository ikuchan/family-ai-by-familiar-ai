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


# ---- タイマーは調停（軽量LLM）が自分で掛ける（知-n・2026-09-15 夜） -------------------


def test_the_arbiter_can_set_a_timer_with_a_tool_input():
    d = _parse(
        '{"branch":"action","action":"set_timer","tool_input":{"after_minutes":3,"label":"パスタ"},"text":"3分ね、測るよ"}',
        can_see=False,
        extra_actions=("set_timer", "cancel_timer", "start_stopwatch"),
    )
    assert d is not None and d.branch == "action" and d.action == "set_timer"
    assert d.tool_input == {"after_minutes": 3, "label": "パスタ"} and d.query == "パスタ"
    assert d.text == ""  # 道具は即返るのでつなぎは言わない（返りを見て 1 回だけ言う）
    d = _parse(
        '{"branch":"action","action":"cancel_timer","tool_input":{"id":"all"}}',
        can_see=False,
        extra_actions=("set_timer", "cancel_timer", "start_stopwatch"),
    )
    assert (
        d is not None
        and d.action == "cancel_timer"
        and d.tool_input == {"id": "all"}
        and d.query == "all"
    )
    # 候補に無ければ判定できない扱い（None → full へ倒れる・従来どおり）
    d = _parse(
        '{"branch":"action","action":"set_timer","tool_input":{"after_minutes":3,"label":"x"}}',
        can_see=False,
        extra_actions=(),
    )
    assert d is None


def test_timer_actions_are_offered_to_the_arbiter_when_the_tool_exists():
    from unittest.mock import MagicMock

    ip = _ip("ゆうすけ", [])
    ip._agent._timer_tool = MagicMock()
    ip._agent._timer_tool.get_tool_definitions = MagicMock(
        return_value=[{"name": n} for n in ("set_timer", "cancel_timer")]
    )
    ip._agent._stopwatch_tool = MagicMock()  # 別物の器（知-u）
    ip._agent._stopwatch_tool.get_tool_definitions = MagicMock(
        return_value=[{"name": "start_stopwatch"}]
    )
    assert {"set_timer", "start_stopwatch", "cancel_timer"} <= set(ip._extra_actions())
    ip._agent._timer_tool = None
    assert "set_timer" not in ip._extra_actions()


def test_the_decisions_tool_input_reaches_the_dispatch():
    from familiar_agent.loop.arbiter import Decision
    from familiar_agent.loop.event_loop import _tool_input_of

    d = Decision(
        branch="action",
        action="set_timer",
        query="パスタ",
        tool_input={"after_minutes": 3, "label": "パスタ"},
    )
    assert _tool_input_of(d) == {"after_minutes": 3, "label": "パスタ"}
    d = Decision(branch="action", action="family_schedule", query="2")
    assert _tool_input_of(d) == {"days": 2}


def test_a_timer_query_without_tool_input_is_read_into_one():
    """調停が `tool_input` でなく `query` に「1分」「7時」と書いてきても掛かる（実機 23:13）。"""
    from familiar_agent.loop.arbiter import timer_input_from_query

    assert timer_input_from_query("1分") == {"after_minutes": 1.0, "label": "タイマー"}
    assert timer_input_from_query("3分後 パスタ") == {"after_minutes": 3.0, "label": "パスタ"}
    # 「何時に」はアラーム（知-q・2026-09-18）。同じ読みで `at` になり、調停の action は set_alarm へ直る。
    assert timer_input_from_query("7時 起こす") == {"at": "7:00", "label": "起こす"}
    assert timer_input_from_query("21時半") == {"at": "21:30", "label": "アラーム"}
    assert timer_input_from_query("パスタ") is None
    d = _parse(
        '{"branch":"action","action":"set_timer","query":"1分","text":"はい、1分ですね。"}',
        can_see=False,
        extra_actions=("set_timer", "cancel_timer", "start_stopwatch"),
    )
    assert (
        d is not None
        and d.action == "set_timer"
        and d.tool_input == {"after_minutes": 1.0, "label": "タイマー"}
    )
    d = _parse(
        '{"branch":"action","action":"cancel_timer","query":"all"}',
        can_see=False,
        extra_actions=("set_timer", "cancel_timer", "start_stopwatch"),
    )
    assert d is not None and d.tool_input == {"id": "all"}
    d = _parse(
        '{"branch":"action","action":"start_stopwatch","query":"ランニング"}',
        can_see=False,
        extra_actions=("set_timer", "cancel_timer", "start_stopwatch"),
    )
    assert d is not None and d.tool_input == {"label": "ランニング"}
    d = _parse(
        '{"branch":"action","action":"set_timer","query":"7時 起こす"}',
        can_see=False,
        extra_actions=("set_timer", "set_alarm", "cancel_alarm"),
    )
    assert (
        d is not None
        and d.action == "set_alarm"
        and d.tool_input == {"at": "7:00", "label": "起こす"}
    )


def test_an_action_branch_with_only_text_is_a_light_reply():
    """動作が無く text だけの action（「鳴らしていい？」と聞きたかった）は light として扱う（実機 23:14）。"""
    d = _parse('{"branch":"action","text":"鳴らしていい？"}', can_see=False, extra_actions=())
    assert d is not None and d.branch == "light" and d.text == "鳴らしていい？"
    assert _parse('{"branch":"action"}', can_see=False, extra_actions=()) is None


def test_timer_actions_carry_no_filler_because_they_return_at_once():
    """道具が 0.1 秒で返るので、つなぎを言うと返りの一言と重なって 2 回出る（実機 08:59）。"""
    d = _parse(
        '{"branch":"action","action":"start_stopwatch","tool_input":{"label":"x"},"text":"はい、時間を測りますね。"}',
        can_see=False,
        extra_actions=("set_timer", "cancel_timer", "start_stopwatch"),
    )
    assert d is not None and d.action == "start_stopwatch" and d.text == ""
    d = _parse(
        '{"branch":"action","action":"search_deferred","query":"天気","text":"調べてみるね"}',
        can_see=False,
        extra_actions=(),
    )
    assert d is not None and d.text == "調べてみるね"  # 遅い調べものはつなぎを残す


def test_a_json_string_in_query_or_tool_input_is_read_as_the_tool_input():
    """調停（Gemini）は tool_input を JSON の**文字列**として query に書く（実機 08:59〜09:00・4 回すべて）。"""
    d = _parse(
        '{"branch":"action","action":"set_timer","query":"{\\"after_minutes\\": 0.5, \\"label\\": \\"パパのお願い\\"}"}',
        can_see=False,
        extra_actions=("set_timer", "cancel_timer", "start_stopwatch"),
    )
    assert (
        d is not None
        and d.action == "set_timer"
        and d.tool_input == {"after_minutes": 0.5, "label": "パパのお願い"}
    )
    d = _parse(
        '{"branch":"action","action":"cancel_timer","tool_input":"{\\"id\\": \\"all\\"}"}',
        can_see=False,
        extra_actions=("set_timer", "cancel_timer", "start_stopwatch"),
    )
    assert d is not None and d.tool_input == {"id": "all"}
    # 動作を取り違えても、入力が {"id": …} だけなら止める意図（「ストップ」に set_timer と書いた）
    d = _parse(
        '{"branch":"action","action":"set_timer","query":"{\\"id\\": \\"all\\"}"}',
        can_see=False,
        extra_actions=("set_timer", "cancel_timer", "start_stopwatch"),
    )
    assert d is not None and d.action == "cancel_timer" and d.tool_input == {"id": "all"}
