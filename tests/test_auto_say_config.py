"""Tests for AgentConfig.auto_say default (silence-control step 5).

auto_say is ON by default; runtime presence + quiet-hours gates keep it from
speaking into an empty room or during quiet hours. It can be disabled with
FAMILIAR_AUTO_SAY=0 (or false/no).
"""

from __future__ import annotations

from familiar_agent.config import AgentConfig


def test_auto_say_on_by_default(monkeypatch) -> None:
    monkeypatch.delenv("FAMILIAR_AUTO_SAY", raising=False)
    monkeypatch.delenv("FAMILIAR_AUTO", raising=False)
    assert AgentConfig().auto_say is True


def test_auto_say_disabled_with_zero(monkeypatch) -> None:
    monkeypatch.setenv("FAMILIAR_AUTO_SAY", "0")
    assert AgentConfig().auto_say is False


def test_auto_say_disabled_with_false(monkeypatch) -> None:
    monkeypatch.setenv("FAMILIAR_AUTO_SAY", "false")
    assert AgentConfig().auto_say is False


def test_auto_say_stays_on_when_explicitly_enabled(monkeypatch) -> None:
    monkeypatch.setenv("FAMILIAR_AUTO_SAY", "1")
    assert AgentConfig().auto_say is True
