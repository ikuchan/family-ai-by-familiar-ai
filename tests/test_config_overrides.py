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
    for field in ("CameraConfig.password", "TTSConfig.elevenlabs_api_key",
                  "MobilityConfig.api_secret", "CameraConfig.host",
                  "CameraConfig.username"):
        assert is_protected(field) is True, field
        assert save_override(field, "x") is False, field
    assert is_protected(_KEY) is False


def test_an_explicit_env_setting_wins_over_the_agents_adjustment():
    """人が env で明示していれば、内省の調整は効かない（人の設定は変えない）。"""
    assert save_override(_KEY, 0.55) is True
    with patch.dict(os.environ, {"DISTILL_MIN_A0": "0.31"}, clear=True):
        assert MemoryConfig().distill_min_a0 == pytest.approx(0.31)


def test_the_default_is_used_when_nothing_is_adjusted():
    with patch.dict(os.environ, {}, clear=True):
        assert MemoryConfig().distill_min_a0 == pytest.approx(0.47)
