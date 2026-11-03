"""出-dk：発話前の検査に差し戻された言い直しの回で、調べもの（search_deferred）を捨てない（2026-11-03・本人の決定ア）。

2026-11-03 19:14、検査は `no-invented-knowledge`（調べれば分かることは search_deferred で確かめてから話す）で差し戻した。
言い直しの主LLM は say と search_deferred を両方呼んだが、「say だけの回」として search_deferred が捨てられ、
「いま調べて確かめています」だけが声になった（調べものは立っていない）。言い直しの回だけ search_deferred を通す。
上限の回（連鎖を必ず閉じる）と「say で言い直して」の回は、いまどおり say だけ。
"""

from __future__ import annotations

import asyncio

import pytest

from familiar_agent.backends.types import ToolCall, system_text
from tests.test_act_on_decision import _ip as _act_ip
from tests.test_act_on_decision import _run as _act_run
from tests.test_act_on_decision import _turn
from tests.test_window_talk_goes_to_main import _ip_in_window, _names, _run

SAY = ToolCall(id="s", name="say", input={"text": "いま調べて確かめています"})
SEARCH = ToolCall(id="q", name="search_deferred", input={"query": "ベトナム語 こんにちは"})
RECALL = ToolCall(id="r", name="recall", input={"query": "x"})


@pytest.fixture(autouse=True)
def _anthropic_like(monkeypatch):
    from familiar_agent.loop import event_loop

    monkeypatch.setattr(event_loop, "_caches_window", lambda backend: True)


def _outside():
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    return InformationProcessing(_agent(stream_returns=[]))


# ── 1. 窓の中の言い直しで、調べものを残す ─────────────────────────────


def test_a_retry_in_the_window_keeps_the_search(monkeypatch):
    _sent, decided = _run(_ip_in_window(), monkeypatch, calls=[SEARCH, SAY], retried=True)
    assert [tc.name for tc in decided.result.tool_calls] == ["search_deferred", "say"]


def test_a_retry_still_drops_other_tools(monkeypatch):
    """**反証の側**：言い直しでも、思い出しなどほかの道具は捨てる。"""
    _sent, decided = _run(_ip_in_window(), monkeypatch, calls=[RECALL, SAY], retried=True)
    assert [tc.name for tc in decided.result.tool_calls] == ["say"]


@pytest.mark.parametrize(
    "kw",
    [
        {"capped": True},
        {"say_asked": True},
        # 上限の回の言い直しも、調べものは捨てる（連鎖を必ず閉じる）
        {"capped": True, "retried": True},
        {"say_asked": True, "retried": True},
    ],
)
def test_capped_and_say_asked_rounds_still_drop_the_search(monkeypatch, kw):
    """**反証の側**：上限の回と「say で言い直して」の回は、調べものも捨てる。"""
    _sent, decided = _run(_ip_in_window(), monkeypatch, calls=[SEARCH, SAY], **kw)
    assert [tc.name for tc in decided.result.tool_calls] == ["say"]


# ── 2. 窓の外の言い直しに、調べものの道具を渡す ───────────────────────


def test_a_retry_outside_the_window_gets_the_search_tool(monkeypatch):
    sent, _decided = _run(_outside(), monkeypatch, calls=[SAY], retried=True)
    assert _names(sent["tools"]) == ["say", "search_deferred"]


def test_a_capped_round_outside_the_window_gets_only_say(monkeypatch):
    """**反証の側**：上限の回は、窓の外でも say だけ。"""
    sent, _decided = _run(_outside(), monkeypatch, calls=[SAY], capped=True)
    assert _names(sent["tools"]) == ["say"]


# ── 3. 言い直しの回に添える文 ───────────────────────────────────────


def test_a_retry_in_the_window_is_told_it_may_search(monkeypatch):
    from familiar_agent.loop.event_loop import RETRY_NOTE

    sent, _decided = _run(_ip_in_window(), monkeypatch, calls=[SAY], retried=True)
    assert RETRY_NOTE in sent["system"][-1]
    assert "say だけを使う" not in sent["system"][-1]
    assert "search_deferred" in RETRY_NOTE


def test_a_capped_round_in_the_window_keeps_the_say_only_note(monkeypatch):
    """**反証の側**：上限の回は、いまどおり「say だけ」の文。"""
    from familiar_agent.loop.event_loop import RETRY_NOTE, SAY_ONLY_NOTE

    sent, _decided = _run(_ip_in_window(), monkeypatch, calls=[SAY], capped=True)
    assert SAY_ONLY_NOTE in system_text(sent["system"])
    assert RETRY_NOTE not in system_text(sent["system"])


# ── 4. 出す反復：言い直しの調べものを投げ、say は声にしない ───────────


def test_a_retried_search_is_thrown_and_the_say_is_not_spoken():
    ip, _a = _act_ip()
    out = _act_run(ip, _turn(SEARCH, SAY), retried=True)
    assert out == ""
    ip._start_lookup.assert_called_once()
    assert ip._start_lookup.call_args.kwargs.get("action") == "search_deferred"
    ip._speak.assert_not_called()


# ── 見直し 1 回目の指摘 ─────────────────────────────────────────────


def test_a_retry_after_a_say_asked_round_has_only_the_retry_note(monkeypatch):
    """「say で言い直して」の回の system（SAY_ONLY_NOTE つき）を言い直しに使っても、食い違う 2 つの言葉を並べない。"""
    from unittest.mock import AsyncMock

    from familiar_agent.backends.types import TurnResult
    from familiar_agent.loop import event_loop
    from familiar_agent.loop.event_loop import RETRY_NOTE, SAY_ONLY_NOTE
    from tests.test_window_talk_goes_to_main import HEAD, L1

    ip = _ip_in_window()
    monkeypatch.setattr(event_loop, "main_llm_cache_on", lambda: True)
    ip._agent.backend.stream_turn = AsyncMock(
        return_value=(TurnResult(stop_reason="tool_use", text="", tool_calls=[SAY]), None)
    )
    system = ("安定", HEAD, L1, f"可変\n\n{SAY_ONLY_NOTE}")
    asyncio.run(
        ip._run_main_llm(
            index=1, messages=[], system=system, effort=None, capped=False,
            memories=[], w_id_map={}, verdict_view="", recent_frame="", max_tokens=100, retried=True,
        )
    )  # fmt: skip
    last = ip._agent.backend.stream_turn.await_args.kwargs["system"][-1]
    assert RETRY_NOTE in last and SAY_ONLY_NOTE not in last


@pytest.mark.parametrize("box", ["spent_actions", "failed_actions"])
def test_a_retry_does_not_search_when_searching_is_spent_or_failed(monkeypatch, box):
    """この求めで調べものを使い切った・失敗したなら、言い直しでも調べものは捨て、say だけの言葉にする。"""
    from familiar_agent.loop.event_loop import RETRY_NOTE, SAY_ONLY_NOTE

    ip = _ip_in_window()
    getattr(ip._req, box).add("search_deferred")
    sent, decided = _run(ip, monkeypatch, calls=[SEARCH, SAY], retried=True)
    assert [tc.name for tc in decided.result.tool_calls] == ["say"]
    assert SAY_ONLY_NOTE in sent["system"][-1] and RETRY_NOTE not in sent["system"][-1]


def test_a_retry_outside_the_window_is_told_it_may_search(monkeypatch):
    from familiar_agent.loop.event_loop import RETRY_NOTE

    sent, _decided = _run(_outside(), monkeypatch, calls=[SAY], retried=True)
    assert RETRY_NOTE in system_text(sent["system"])
