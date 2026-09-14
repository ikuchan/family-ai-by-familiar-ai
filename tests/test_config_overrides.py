"""Config の自己調整の器（記-a）。

設計（`直近の進め方と進捗` v0.14）は内省の1パスに「Config 自己調整（範囲内・人の設定は
変えない）」を含める。ここで作るのは**器だけ**で、内省が値を提案する部分は記-a-に で足す。

優先順位は3段：**env（人が明示）> agent_state（内省が調整）> Config の既定**。
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from familiar_agent.config import MemoryConfig
from familiar_agent.config_overrides import (
    RANGES,
    clear_cache,
    is_protected,
    load_overrides,
    save_override,
)

_KEY = "MemoryConfig.distill_min_a0"


@pytest.fixture(autouse=True)
def _clean():
    """各テストの前後で、保存済みの調整とキャッシュを消す。"""
    from familiar_agent.config_overrides import _delete_all

    _delete_all()
    clear_cache()
    yield
    _delete_all()
    clear_cache()


def test_a_value_in_range_is_saved_and_reaches_the_config():
    assert save_override(_KEY, 0.55) is True
    assert load_overrides()[_KEY] == pytest.approx(0.55)
    with patch.dict(os.environ, {}, clear=True):
        assert MemoryConfig().distill_min_a0 == pytest.approx(0.55)


def test_a_value_outside_the_range_is_rejected():
    """範囲は実測の分布から決めた（p10 0.469・p25 0.604・最小 0.143）。"""
    low, high = RANGES[_KEY]
    assert save_override(_KEY, high + 0.1) is False
    assert save_override(_KEY, low - 0.1) is False
    assert _KEY not in load_overrides()


def test_an_unregistered_field_is_rejected():
    """範囲を登録していない値は、内省が変えられない（安全側）。"""
    assert save_override("DriveConfig.rate", 0.5) is False
    assert save_override("MemoryConfig.recall_k", 9) is False


def test_connection_settings_are_protected():
    """接続情報は内省に触らせない。壊れると機器へ繋がらず、復旧に人手が要る。"""
    for field in (
        "CameraConfig.password",
        "TTSConfig.elevenlabs_api_key",
        "MobilityConfig.api_secret",
        "CameraConfig.host",
        "CameraConfig.username",
    ):
        assert is_protected(field) is True, field
        assert save_override(field, "x") is False, field
    assert is_protected(_KEY) is False


def test_the_env_is_not_read_for_registered_settings():
    """登録した設定値は **DB > 既定**。`.env` は読まない（4 層の外形・2026-09-14・記-a-に）。

    以前は「env > DB > 既定」だった。`.env` は機密と機体固有のものだけを持ち、4 層の値は
    置かない。同名の環境変数があっても使わない（WARNING を出す）。
    """
    from familiar_agent import config_overrides as co

    assert save_override(_KEY, 0.55) is True
    co.clear_cache()
    with patch.dict(os.environ, {"DISTILL_MIN_A0": "0.31"}, clear=True):
        assert MemoryConfig().distill_min_a0 == pytest.approx(0.55)


def test_the_arbiter_timeout_is_a_registered_setting():
    from familiar_agent import config_overrides as co
    from familiar_agent.config import AgentConfig

    assert RANGES["AgentConfig.arbiter_timeout_sec"] == (1.0, 10.0)
    co._delete_all()
    co.clear_cache()
    with patch.dict(os.environ, {"ARBITER_TIMEOUT_SEC": "9.0"}, clear=True):
        assert AgentConfig().arbiter_timeout_sec == pytest.approx(5.0)  # env は読まない
    assert save_override("AgentConfig.arbiter_timeout_sec", 6.0)
    co.clear_cache()
    assert AgentConfig().arbiter_timeout_sec == pytest.approx(6.0)
    co._delete_all()
    co.clear_cache()


def test_there_is_one_resolver_and_it_has_no_env_branch():
    import inspect

    from familiar_agent import config as cfg_mod
    from familiar_agent import config_overrides as co

    assert not hasattr(co, "resolve_float") and not hasattr(cfg_mod, "_resolve_float")
    src = inspect.getsource(cfg_mod._resolve_setting)
    assert "float(os.environ[" not in src


def test_the_default_is_used_when_nothing_is_adjusted():
    with patch.dict(os.environ, {}, clear=True):
        assert MemoryConfig().distill_min_a0 == pytest.approx(0.47)


# ── 内部状態の言葉の境目は層 3 の設定値（情-f・2026-09-14） ────────────────────


def test_the_inner_state_boundaries_are_registered_with_ranges():
    """気分 4 軸 × 4 分位＝16 個と、欲求 5 軸の p70＝5 個が、範囲つきで登録されている。"""
    from familiar_agent.config_overrides import is_protected

    keys = [k for k in RANGES if k.startswith("InnerStateConfig.")]
    assert len(keys) == 21, keys
    for k in keys:
        lo, hi = RANGES[k]
        assert 0.0 <= lo < hi <= 1.0, k
        assert not is_protected(k), k


def test_the_inner_state_config_reads_db_over_default_and_ignores_env(monkeypatch):
    """4 層の外形：登録した設定値は DB > 既定。`.env` は読まない（あれば WARNING を出して使わない）。"""

    from familiar_agent import config_overrides as co
    from familiar_agent.config import InnerStateConfig

    co.clear_cache()
    co._delete_all()
    assert InnerStateConfig().mood_p == (0.10, 0.10, 0.25, 0.35)
    assert co.save_override("InnerStateConfig.mood_p_p70", 0.30)
    co.clear_cache()
    assert InnerStateConfig().mood_p == (0.10, 0.10, 0.30, 0.35)
    # env に同名があっても使わない。
    monkeypatch.setenv("INNER_STATE_MOOD_P_P70", "0.99")
    co.clear_cache()
    assert InnerStateConfig().mood_p[2] == 0.30
    co._delete_all()
    co.clear_cache()
