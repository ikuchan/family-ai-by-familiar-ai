"""静穏時間の規則（`routines.py`）。

継続制御の検査は 041 で撤去した。`HeartbeatRuntime` は環-c で呼び出し側を失って
おり、本番からの呼び出しが全メソッド0件だった。
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from familiar_agent.routines import QuietHoursRule, evaluate_routine_state, quiet_hours_rule


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
