"""Tests for EmbodiedAgent.should_deliver_deferred_result().

Covers the combined gating logic across search + fetch deferred tools.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from familiar_agent.agent import EmbodiedAgent


def _make_deferred(
    *,
    has_pending: bool = False,
    is_running: bool = False,
    user_initiated: bool = False,
) -> MagicMock:
    m = MagicMock()
    m.has_pending = has_pending
    m.is_running = is_running
    m.has_user_initiated_pending = user_initiated
    return m


def _make_agent_stub(
    *,
    search_pending: bool = False,
    search_running: bool = False,
    fetch_pending: bool = False,
    fetch_running: bool = False,
    search_user_initiated: bool = False,
    fetch_user_initiated: bool = False,
    presence: float = 1.0,
    quiet: bool = False,
    social_act: str | None = None,
    user_recent: bool = True,
) -> EmbodiedAgent:
    import time
    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    agent._deferred_search = _make_deferred(has_pending=search_pending, is_running=search_running, user_initiated=search_user_initiated)
    agent._deferred_fetch = _make_deferred(has_pending=fetch_pending, is_running=fetch_running, user_initiated=fetch_user_initiated)
    agent._social_presence_permission = MagicMock(return_value=presence)
    agent._last_human_at = time.time() if user_recent else 0.0
    agent._schedule_rule = None
    if quiet:
        rule = MagicMock()
        rule.is_quiet = MagicMock(return_value=True)
        agent._schedule_rule = rule
    if social_act is not None:
        policy = SimpleNamespace(primary_act=social_act)
        agent._last_social_decision = policy
    else:
        agent._last_social_decision = None
    return agent


# ---------------------------------------------------------------------------
# Gate 1: pending checks
# ---------------------------------------------------------------------------


def test_returns_false_when_nothing_pending():
    agent = _make_agent_stub()
    assert agent.should_deliver_deferred_result() is False


def test_returns_true_when_search_pending_only():
    agent = _make_agent_stub(search_pending=True)
    assert agent.should_deliver_deferred_result() is True


def test_returns_true_when_fetch_pending_only():
    agent = _make_agent_stub(fetch_pending=True)
    assert agent.should_deliver_deferred_result() is True


def test_returns_true_when_both_pending():
    agent = _make_agent_stub(search_pending=True, fetch_pending=True)
    assert agent.should_deliver_deferred_result() is True


# ---------------------------------------------------------------------------
# Gate 1b: wait while running
# ---------------------------------------------------------------------------


def test_waits_while_search_still_running():
    agent = _make_agent_stub(search_pending=True, search_running=True)
    assert agent.should_deliver_deferred_result() is False


def test_waits_while_fetch_still_running():
    agent = _make_agent_stub(search_pending=True, fetch_running=True)
    assert agent.should_deliver_deferred_result() is False


def test_waits_when_search_done_but_fetch_still_running():
    """Search finished and has results, but fetch is still in flight — hold."""
    agent = _make_agent_stub(search_pending=True, fetch_running=True)
    assert agent.should_deliver_deferred_result() is False


def test_delivers_when_both_complete():
    agent = _make_agent_stub(search_pending=True, fetch_pending=True,
                             search_running=False, fetch_running=False)
    assert agent.should_deliver_deferred_result() is True


def test_delivers_when_only_fetch_complete_and_search_never_started():
    """No search was queued; only fetch results are ready."""
    agent = _make_agent_stub(fetch_pending=True,
                             search_running=False, fetch_running=False)
    assert agent.should_deliver_deferred_result() is True


# ---------------------------------------------------------------------------
# Gate 2: presence
# ---------------------------------------------------------------------------


def test_blocks_when_no_one_present():
    agent = _make_agent_stub(search_pending=True, presence=0.0)
    assert agent.should_deliver_deferred_result() is False


def test_allows_when_person_present():
    agent = _make_agent_stub(search_pending=True, presence=1.0)
    assert agent.should_deliver_deferred_result() is True


# ---------------------------------------------------------------------------
# Gate 3: quiet hours
# ---------------------------------------------------------------------------


def test_blocks_during_quiet_hours():
    agent = _make_agent_stub(search_pending=True, quiet=True)
    assert agent.should_deliver_deferred_result() is False


def test_no_schedule_rule_does_not_block():
    agent = _make_agent_stub(search_pending=True)
    agent._schedule_rule = None
    assert agent.should_deliver_deferred_result() is True


# ---------------------------------------------------------------------------
# Gate 4: social policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("act", [
    "grief_signal", "venting", "fatigue_signal", "repair_attempt", "boundary_assertion",
])
def test_blocks_on_sensitive_social_acts(act: str):
    agent = _make_agent_stub(search_pending=True, social_act=act)
    assert agent.should_deliver_deferred_result() is False


def test_allows_on_neutral_social_act():
    agent = _make_agent_stub(search_pending=True, social_act="casual_chat")
    assert agent.should_deliver_deferred_result() is True


def test_no_prior_social_decision_does_not_block():
    agent = _make_agent_stub(search_pending=True)
    assert agent._last_social_decision is None
    assert agent.should_deliver_deferred_result() is True


# ---------------------------------------------------------------------------
# Early-exit: missing _deferred_search
# ---------------------------------------------------------------------------


def test_returns_false_when_deferred_search_not_initialized():
    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    # _deferred_search intentionally absent
    assert agent.should_deliver_deferred_result() is False


# ---------------------------------------------------------------------------
# Gate 3: quiet hours bypass for user-initiated searches
# ---------------------------------------------------------------------------


def test_user_initiated_search_bypasses_quiet_hours():
    agent = _make_agent_stub(search_pending=True, quiet=True, search_user_initiated=True)
    assert agent.should_deliver_deferred_result() is True


def test_user_initiated_fetch_bypasses_quiet_hours():
    agent = _make_agent_stub(fetch_pending=True, quiet=True, fetch_user_initiated=True)
    assert agent.should_deliver_deferred_result() is True


def test_autonomous_search_still_blocked_during_quiet_hours():
    agent = _make_agent_stub(search_pending=True, quiet=True, search_user_initiated=False)
    assert agent.should_deliver_deferred_result() is False


def test_autonomous_fetch_still_blocked_during_quiet_hours():
    agent = _make_agent_stub(fetch_pending=True, quiet=True, fetch_user_initiated=False)
    assert agent.should_deliver_deferred_result() is False


def test_mixed_user_and_autonomous_pending_bypasses_quiet_hours():
    # One user-initiated + one autonomous pending → bypass (user's request takes priority)
    agent = _make_agent_stub(
        search_pending=True, search_user_initiated=True,
        fetch_pending=True, fetch_user_initiated=False,
        quiet=True,
    )
    assert agent.should_deliver_deferred_result() is True


def test_user_initiated_search_does_not_bypass_when_user_not_recent():
    # User-initiated but user hasn't been active for >30 min → no bypass
    agent = _make_agent_stub(
        search_pending=True, quiet=True, search_user_initiated=True, user_recent=False,
    )
    assert agent.should_deliver_deferred_result() is False


def test_user_initiated_fetch_does_not_bypass_when_user_not_recent():
    agent = _make_agent_stub(
        fetch_pending=True, quiet=True, fetch_user_initiated=True, user_recent=False,
    )
    assert agent.should_deliver_deferred_result() is False
