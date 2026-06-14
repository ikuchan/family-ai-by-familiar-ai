"""Tests for decay settings in MemoryConfig (config.py)."""
from __future__ import annotations

import importlib


def _fresh_config(monkeypatch, **env_vars):
    """Import MemoryConfig with env vars set, bypassing module-level caching."""
    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)
    import familiar_agent.config as cfg_mod
    importlib.reload(cfg_mod)
    return cfg_mod.MemoryConfig()


def test_memory_config_has_decay_settings(monkeypatch):
    monkeypatch.setenv("RECALL_HALF_LIFE_DAYS", "5.0")
    monkeypatch.setenv("RECALL_TIME_FLOOR", "0.3")
    from familiar_agent.config import MemoryConfig
    cfg = MemoryConfig()
    assert cfg.recall_half_life_days == 5.0
    assert cfg.recall_time_floor == 0.3


def test_memory_config_recall_min_score(monkeypatch):
    monkeypatch.setenv("RECALL_MIN_SCORE", "0.6")
    from familiar_agent.config import MemoryConfig
    cfg = MemoryConfig()
    assert cfg.recall_min_score == pytest.approx(0.6)


def test_memory_config_defaults():
    """デフォルト値の確認（envなし）。"""
    from familiar_agent.config import MemoryConfig
    cfg = MemoryConfig()
    assert cfg.recall_half_life_days == 7.0
    assert cfg.recall_time_floor == 0.25
    assert cfg.recall_min_score == 0.0


def test_memory_config_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("RECALL_HALF_LIFE_DAYS", "not-a-float")
    from familiar_agent.config import MemoryConfig
    cfg = MemoryConfig()
    assert cfg.recall_half_life_days == 7.0


import pytest
