"""Tests for the TUI real-time status line formatting helpers.

The status line shows:  ⠋ 23s · ↓ 6.5k  (or  ⠋ 1m 23s · ↓ 6.5k)
during agent processing, updating every 0.08 s.
"""

from __future__ import annotations


from familiar_agent.tui import _format_elapsed


# ── _format_elapsed ────────────────────────────────────────────────────────


class TestFormatElapsed:
    def test_zero_seconds(self):
        assert _format_elapsed(0.0) == "0s"

    def test_fractional_rounds_down(self):
        assert _format_elapsed(0.9) == "0s"

    def test_single_digit_seconds(self):
        assert _format_elapsed(7.0) == "7s"

    def test_double_digit_seconds(self):
        assert _format_elapsed(45.0) == "45s"

    def test_59_seconds(self):
        assert _format_elapsed(59.9) == "59s"

    def test_exactly_60_seconds_shows_minutes(self):
        assert _format_elapsed(60.0) == "1m 00s"

    def test_63_seconds(self):
        assert _format_elapsed(63.0) == "1m 03s"

    def test_7m_43s(self):
        assert _format_elapsed(7 * 60 + 43) == "7m 43s"

    def test_hour_plus(self):
        """Over an hour still shown as minutes."""
        assert _format_elapsed(3661.0) == "61m 01s"
