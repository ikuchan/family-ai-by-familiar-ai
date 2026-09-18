"""確認を機械の状態にする（出-y・2026-09-18・`設計方針_タイマー` v0.9 §10・`設計方針_アラーム` v0.2）。

実機 15:42（id=13）で主LLM が `confirmed=true` を勝手に付けて掛けた。確認は求め（LLM の引数）でなく、
**機械の状態**（`agent._pending_confirm`・`core/confirm_state.py`）が持つ。

- 道具が「確かめて」と判定したら、預かり（何を・どの入力で・いつ・何と聞いたか）を置き、返りは確認文だけ。
  `confirmed` は道具の定義に無く、LLM が入力に書いても効かない。
- 次の求めで預かりが生きていれば（寿命 `CONFIRM_TTL_SEC`・300 秒〔仮〕）、W の最上部に `[確認待ち]`、
  候補に `confirm`／`decline`。`confirm` → 機械が預かった入力で掛ける（帰りは「掛けた」）。
  `decline` → 捨てて「やめた」。別の `set_timer` → 預かりは上書き。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.core import confirm_state
from familiar_agent.loop import workspace
from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_alarm import _tool as _alarm_tool
from tests.test_event_loop import _agent
from tests.test_timer_tool import _tool as _timer_tool

T0 = 10_000.0


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    monkeypatch.setenv("TIMER_SILENCE", "true")
    monkeypatch.setenv("TIMER_MIC_CLOSE", "true")
    monkeypatch.setenv("TIMER_CONFIRM", "true")


# ── 道具：預かりを置き、確認文だけ返す ─────────────────────────────────────────


def test_set_timer_leaves_a_pending_confirm_and_no_hint_to_call_again():
    t, store, _ = _timer_tool()
    asked = []
    t._ask = lambda pc: asked.append(pc)
    text, ok = asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}))
    assert ok and store.active() == [] and "confirmed" not in text and "3 分" in text
    (pc,) = asked
    assert pc.action == "set_timer" and pc.tool_input == {"after_minutes": 3, "label": "パスタ"}
    assert "3 分" in pc.text and "パスタ" in pc.what


def test_confirmed_in_the_input_is_ignored_but_the_machine_keyword_sets():
    t, store, _ = _timer_tool()
    t._ask = lambda pc: None
    asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ", "confirmed": True}))
    assert store.active() == []  # LLM が書いても効かない
    text, ok = asyncio.run(
        t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}, confirmed=True)
    )
    assert ok and len(store.active()) == 1 and "id=1" in text


def test_the_definitions_have_no_confirmed():
    t, _, _ = _timer_tool()
    a, _, _, _ = _alarm_tool()
    for d in t.get_tool_definitions() + a.get_tool_definitions():
        assert (
            "confirmed" not in d["input_schema"]["properties"]
            and "confirmed" not in d["description"]
        )


def test_set_alarm_in_quiet_hours_leaves_a_pending_confirm():
    t, store, _, _ = _alarm_tool()
    asked = []
    t._ask = lambda pc: asked.append(pc)
    text, ok = asyncio.run(t.call("set_alarm", {"at": "6:30", "label": "起こす"}))
    assert ok and store.active() == [] and "confirmed" not in text
    assert asked[0].action == "set_alarm" and asked[0].tool_input["at"] == "6:30"
    asyncio.run(t.call("set_alarm", {"at": "6:30", "label": "起こす"}, confirmed=True))
    assert len(store.active()) == 1


# ── 状態：寿命と枠 ────────────────────────────────────────────────────────────


def _pc(asked_at=T0):
    return confirm_state.PendingConfirm(
        action="set_timer",
        tool_input={"after_minutes": 3, "label": "パスタ"},
        asked_at=asked_at,
        text="3 分のタイマーね、いい？",
        what="タイマーを掛ける「パスタ」（3 分）",
    )


def test_a_pending_confirm_lives_for_the_ttl_then_dies():
    assert confirm_state.alive(_pc(), now=T0 + 299, ttl=300)
    assert not confirm_state.alive(_pc(), now=T0 + 301, ttl=300)
    assert not confirm_state.alive(None, now=T0, ttl=300)


def test_the_frame_names_what_and_the_question_and_the_two_answers():
    f = confirm_state.frame(_pc())
    assert (
        f.startswith("[確認待ち]")
        and "パスタ" in f
        and "いい？" in f
        and "confirm" in f
        and "decline" in f
    )


# ── 装置：候補・W の最上部・confirm／decline の実行 ──────────────────────────


def _ip(pending):
    a = _agent(stream_returns=[])
    a._pending_confirm = pending
    a.confirm_alive = MagicMock(return_value=pending is not None)
    a.confirm_frame = MagicMock(return_value=confirm_state.frame(pending) if pending else "")
    a._timer_tool = MagicMock()
    a._timer_tool.call = AsyncMock(return_value=("掛けた：id=1 「パスタ」 10:03 に鳴る", True))
    a._timer_tool.get_tool_definitions = MagicMock(return_value=[])
    a._alarm_tool = None
    a._mcp = None
    ip = InformationProcessing(a)
    ip._gated = lambda defs: defs
    return a, ip


def test_confirm_and_decline_are_offered_only_while_a_confirm_is_pending():
    _, ip = _ip(_pc())
    acts = set(ip._extra_actions())
    assert {"confirm", "decline"} <= acts
    _, ip = _ip(None)
    assert not {"confirm", "decline"} & set(ip._extra_actions())


def test_the_request_carries_the_frame_and_w_puts_it_on_top():
    _, ip = _ip(_pc())
    asyncio.run(ip._begin_request(kind="発話", text="いいよ", utterance="いいよ"))
    assert ip._req.confirm_frame.startswith("[確認待ち]")
    oif = MagicMock()
    oif.latest_origins = MagicMock(return_value=[])
    ws = workspace.Workspace.build(oif, [], ip._req, n_arbiter=2, n_main=3)
    assert ws.render(2).startswith("[確認待ち]")
    _, ip2 = _ip(None)
    asyncio.run(ip2._begin_request(kind="発話", text="やあ", utterance="やあ"))
    assert ip2._req.confirm_frame == ""


def test_confirm_calls_the_tool_with_the_kept_input_and_the_machine_keyword():
    a, ip = _ip(_pc())
    a.resolve_confirm = AsyncMock(return_value=("掛けた：id=1", True))
    asyncio.run(ip._begin_request(kind="発話", text="いいよ", utterance="いいよ"))
    asyncio.run(ip._run_lookup_body("confirm", {}, "「いい」と言われて掛ける", None, 1))
    a.resolve_confirm.assert_awaited_once_with(True, now=ip._req.began_at)
    trig = ip._triggers.get_nowait()
    assert trig.kind == "完了" and trig.result.startswith("掛けた") and not trig.failed


def test_decline_is_also_a_return_of_the_machine():
    a, ip = _ip(_pc())
    a.resolve_confirm = AsyncMock(return_value=("やめた：「パスタ」は掛けない", True))
    asyncio.run(ip._begin_request(kind="発話", text="やめて", utterance="やめて"))
    asyncio.run(ip._run_lookup_body("decline", {}, "やめる", None, 1))
    a.resolve_confirm.assert_awaited_once_with(False, now=ip._req.began_at)
    assert (
        "confirm" in workspace.RETURN_WITHOUT_RECALL
        and "decline" in workspace.RETURN_WITHOUT_RECALL
    )


# ── agent：預かりを解く ─────────────────────────────────────────────────────


def _bare_agent():
    from familiar_agent.agent import EmbodiedAgent

    a = EmbodiedAgent.__new__(EmbodiedAgent)
    a._pending_confirm = None
    a._timer_tool = MagicMock()
    a._timer_tool.call = AsyncMock(return_value=("掛けた：id=1", True))
    a._alarm_tool = MagicMock()
    a._alarm_tool.call = AsyncMock(return_value=("掛けた：id=2", True))
    a.config = MagicMock()
    a.config.confirm_ttl_sec = 300.0
    return a


def test_the_agent_keeps_one_pending_and_the_newer_one_wins(monkeypatch):
    a = _bare_agent()
    monkeypatch.setattr("familiar_agent.agent.time.time", lambda: T0)
    a.ask_confirm(_pc())
    a.ask_confirm(_pc(asked_at=T0 + 1))
    assert a._pending_confirm.asked_at == T0 + 1 and a.confirm_alive()
    monkeypatch.setattr("familiar_agent.agent.time.time", lambda: T0 + 400)
    assert not a.confirm_alive() and a.confirm_frame() == ""


def test_yes_calls_the_tool_and_clears_no_just_clears(monkeypatch):
    a = _bare_agent()
    monkeypatch.setattr("familiar_agent.agent.time.time", lambda: T0)
    a.ask_confirm(_pc())
    text, ok = asyncio.run(a.resolve_confirm(True, now="t"))
    a._timer_tool.call.assert_awaited_once_with(
        "set_timer", {"after_minutes": 3, "label": "パスタ"}, now="t", confirmed=True
    )
    assert ok and text.startswith("掛けた") and a._pending_confirm is None
    a.ask_confirm(_pc())
    text, ok = asyncio.run(a.resolve_confirm(False, now="t"))
    assert ok and "やめ" in text and a._pending_confirm is None
    text, ok = asyncio.run(a.resolve_confirm(True, now="t"))
    assert not ok  # 預かりが無い


def test_an_alarm_pending_goes_to_the_alarm_tool(monkeypatch):
    a = _bare_agent()
    monkeypatch.setattr("familiar_agent.agent.time.time", lambda: T0)
    pc = confirm_state.PendingConfirm(
        action="set_alarm",
        tool_input={"at": "6:30", "label": "起こす"},
        asked_at=T0,
        text="q",
        what="w",
    )
    a.ask_confirm(pc)
    asyncio.run(a.resolve_confirm(True, now="t"))
    a._alarm_tool.call.assert_awaited_once_with(
        "set_alarm", {"at": "6:30", "label": "起こす"}, confirmed=True
    )
