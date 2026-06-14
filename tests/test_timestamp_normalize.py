"""Tests for Issue A2: observations timestamp normalization.

After migration 016:
- observations.timestamp is TIMESTAMPTZ (not TEXT)
- date and time columns are dropped
- memory.py derives date/time strings from timestamp in Python

All public API return types are unchanged (date/time remain as YYYY-MM-DD / HH:MM strings).
"""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import patch

import pytest

from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel


@pytest.fixture()
def memory():
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[1.0, 0.0, 0.0]]),
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]),
    ):
        yield ObservationMemory()


@pytest.fixture()
def memory_with_data(memory):
    memory.save("テスト記憶の内容", kind="observation", emotion="curious")
    return memory


def test_recall_returns_date_time_derived_from_timestamp(memory_with_data):
    """date/time カラム廃止後も返り値に date と time が含まれること。"""
    results = memory_with_data.recall("テスト", n=5)
    for r in results:
        assert "date" in r, f"date key missing: {r.keys()}"
        assert "time" in r, f"time key missing: {r.keys()}"
        # date は YYYY-MM-DD 形式の文字列であること
        datetime.strptime(r["date"], "%Y-%m-%d")
        # time は HH:MM 形式の文字列であること
        assert len(r["time"]) == 5 and r["time"][2] == ":", f"time format wrong: {r['time']!r}"


def test_get_dates_with_observations_returns_date_strings(memory_with_data):
    """日付リストが従来通り YYYY-MM-DD 文字列で返ること（呼び出し側互換）。"""
    dates = memory_with_data.get_dates_with_observations(days=7)
    assert all(isinstance(d, str) for d in dates)
    for d in dates:
        datetime.strptime(d, "%Y-%m-%d")


def test_get_observations_for_date_filters_correctly(memory_with_data):
    """timestamp::date での日付絞り込みが正しく動くこと。"""
    today = date.today().isoformat()
    obs = memory_with_data.get_observations_for_date(today, limit=50)
    assert isinstance(obs, list)
    # 今日保存した記憶が含まれていること
    assert len(obs) > 0
    for o in obs:
        assert "time" in o
        assert len(o["time"]) == 5 and o["time"][2] == ":"


def test_save_and_recall_roundtrip(memory):
    """date/time カラムなしで保存 → 想起が一貫すること。"""
    memory.save("正規化テスト記憶", emotion="neutral", kind="conversation")
    results = memory.recall("正規化テスト", n=3)
    assert any("正規化テスト" in r.get("summary", "") for r in results)


def test_recent_feelings_returns_date_time(memory):
    """recent_feelings() の返り値に date/time が含まれること。"""
    memory.save("楽しかった今日", kind="feeling", emotion="happy")
    feelings = memory.recent_feelings(n=5)
    for f in feelings:
        assert "date" in f
        assert "time" in f
        datetime.strptime(f["date"], "%Y-%m-%d")


def test_get_earliest_date_returns_string_or_none(memory_with_data):
    """get_earliest_date() が YYYY-MM-DD 文字列または None を返すこと。"""
    result = memory_with_data.get_earliest_date()
    assert result is None or (isinstance(result, str) and len(result) == 10)
