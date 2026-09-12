"""表示先で例外が出たら、握りつぶさずログに残す（2026-09-12 実機で露見）。

`_emit` は GUI の表示関数を呼ぶ。そこで落ちても発話は止めないが、**何も残さない**と
「表示関数に渡したのに画面に無い」を追えない。
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from familiar_agent.loop.event_loop import InformationProcessing


def _loop_with_outputs(on_text, on_action) -> InformationProcessing:
    ip = InformationProcessing.__new__(InformationProcessing)
    ip._on_text = on_text
    ip._on_action = on_action
    return ip


def test_a_failing_display_is_logged_not_swallowed(caplog) -> None:
    on_action = MagicMock(side_effect=RuntimeError("wrapped C/C++ object has been deleted"))
    ip = _loop_with_outputs(None, on_action)
    with caplog.at_level(logging.WARNING, logger="familiar_agent.loop.event_loop"):
        ip._emit("目の前を見ますね。")  # 落ちない
    msgs = [r.getMessage() for r in caplog.records]
    assert any("表示先" in m and "目の前を見ますね。" in m for m in msgs), msgs
    assert any(r.exc_info for r in caplog.records), "例外の中身（トレース）が残らない"


def test_a_working_display_logs_nothing(caplog) -> None:
    ip = _loop_with_outputs(None, MagicMock())
    with caplog.at_level(logging.WARNING, logger="familiar_agent.loop.event_loop"):
        ip._emit("目の前を見ますね。")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
