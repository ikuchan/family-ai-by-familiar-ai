from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from familiar_agent.heartbeat import HeartbeatRuntime
from familiar_agent.routines import QuietHoursRule, evaluate_routine_state, quiet_hours_rule
from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel


def test_quiet_hours_suppress_intrusive_actions() -> None:
    rule = QuietHoursRule(start_hour=22, end_hour=7)
    decision = evaluate_routine_state(rule, datetime(2026, 4, 15, 23, 30))

    assert decision.quiet_hours is True
    assert decision.schedule_multiplier < 1.0


def test_quiet_hours_come_from_the_environment_then_config() -> None:
    """出所は環境変数 → Config の既定の2段だけ。

    以前は `~/.familiar_ai/schedule.conf` と作業ディレクトリの `ROUTINES.md` も読んでいたが、
    どちらも存在せずコード内の既定が効いているだけだった。出所が4段あると、どの値が
    効いているのかを確かめるのに4箇所を見ることになる。
    """
    import os

    with patch.dict(os.environ, {"QUIET_HOURS_START": "21", "QUIET_HOURS_END": "5"}, clear=True):
        rule = quiet_hours_rule()
    assert (rule.start_hour, rule.end_hour) == (21, 5)
    assert rule.is_quiet(datetime(2026, 6, 11, 22, 0)) is True
    assert rule.is_quiet(datetime(2026, 6, 11, 10, 0)) is False


def test_quiet_hours_fall_back_to_the_config_default() -> None:
    import os

    with patch.dict(os.environ, {}, clear=True):
        rule = quiet_hours_rule()
    assert (rule.start_hour, rule.end_hour) == (23, 7)


def test_continuation_chain_carries_over_and_stops_at_max_depth(tmp_path: Path) -> None:
    db_path = tmp_path / "observations.db"
    with patch.object(_EmbeddingModel, "pre_warm"):
        memory = ObservationMemory(db_path=str(db_path))
        runtime = HeartbeatRuntime(memory=memory, quiet_rule=QuietHoursRule(), max_chain_depth=3)

        assert runtime.apply_status("CONTINUE:step-1").status == "CONTINUE:step-1"
        assert runtime.apply_status("CONTINUE:step-2").status == "CONTINUE:step-2"
        assert runtime.apply_status("CONTINUE:step-3").status == "CONTINUE:step-3"
        overflow = runtime.apply_status("CONTINUE:step-4")

        assert overflow.status == "DEFER:step-4"
        assert overflow.persisted_remainder is True
        open_items = memory.list_unfinished_business()
        assert len(open_items) == 1
        assert open_items[0]["summary"] == "step-4"
        memory.close()


def test_heartbeat_persists_continuation_state_across_restarts() -> None:
    runtime = HeartbeatRuntime(
        quiet_rule=QuietHoursRule(),
        max_chain_depth=3,
    )
    runtime.apply_status("CONTINUE:follow-up tomorrow")

    restored = HeartbeatRuntime(
        quiet_rule=QuietHoursRule(),
        max_chain_depth=3,
    )

    assert "follow-up tomorrow" in restored.continuity_context_for_prompt()
