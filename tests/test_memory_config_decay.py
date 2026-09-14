"""Tests for decay settings in MemoryConfig (config.py)."""

from __future__ import annotations

import importlib

import pytest


def _fresh_config(monkeypatch, **env_vars):
    """Import MemoryConfig with env vars set, bypassing module-level caching."""
    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)
    import familiar_agent.config as cfg_mod

    importlib.reload(cfg_mod)
    return cfg_mod.MemoryConfig()


def test_memory_config_has_decay_settings(monkeypatch):
    monkeypatch.setenv("RECALL_TIME_FLOOR", "0.3")
    from familiar_agent.config import MemoryConfig

    cfg = MemoryConfig()
    assert cfg.recall_time_floor == 0.3


def test_the_half_life_is_a_layer_three_setting_not_an_env(monkeypatch):
    """$HL$ は層 3 の設定値（DB > 既定・`.env` を読まない・記-a-ろ-い・2026-09-14）。既定 10 日。

    「昨日のことは必ず思い出す・10 日前はあまり思い出さない」（`出来事を畳む` v0.1）。
    """
    from familiar_agent import config_overrides as co
    from familiar_agent.config import MemoryConfig

    co.clear_cache()
    assert MemoryConfig().recall_half_life_days == 10.0
    assert co.RANGES["MemoryConfig.recall_half_life_days"] == (1.0, 30.0)
    monkeypatch.setenv("RECALL_HALF_LIFE_DAYS", "3.0")
    co.clear_cache()
    assert MemoryConfig().recall_half_life_days == 10.0  # env は読まない


def test_memory_config_recall_min_score(monkeypatch):
    monkeypatch.setenv("RECALL_MIN_SCORE", "0.6")
    from familiar_agent.config import MemoryConfig

    cfg = MemoryConfig()
    assert cfg.recall_min_score == pytest.approx(0.6)


def test_memory_config_defaults():
    """デフォルト値の確認（envなし）。HL は 10 日（記-a-ろ-い・2026-09-14）、t_floor は課題5 v0.24。"""
    from familiar_agent.config import MemoryConfig

    cfg = MemoryConfig()
    assert cfg.recall_half_life_days == 10.0
    assert cfg.recall_time_floor == 0.001
    assert cfg.recall_min_score == pytest.approx(0.05)
    assert cfg.recall_primary_n == 50


def test_memory_config_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("RECALL_MIN_SCORE", "not-a-float")
    from familiar_agent.config import MemoryConfig

    cfg = MemoryConfig()
    assert cfg.recall_min_score == pytest.approx(0.05)


def test_recent_exchange_windows_are_two_and_come_from_env(monkeypatch):
    """直近のやりとりの窓 n は軽量LLM と主LLM で別々に持つ（記-h・`課題5` D 章）。

    作り方は同じで、窓の大きさだけが違う。既定は 3 と 6（2026-09-13 決定）。
    """
    cfg = _fresh_config(monkeypatch)
    assert (cfg.recent_exchanges_arbiter, cfg.recent_exchanges_main) == (3, 6)
    cfg = _fresh_config(monkeypatch, RECENT_EXCHANGES_ARBITER="2", RECENT_EXCHANGES_MAIN="8")
    assert (cfg.recent_exchanges_arbiter, cfg.recent_exchanges_main) == (2, 8)
