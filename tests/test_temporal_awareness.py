"""Tests for temporal / anniversary awareness.

Phase 4 of companion-likeness Round 2.
The agent knows "on this day" events and milestones, surfacing them in morning reconstruction.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta
from unittest.mock import patch

import pytest

from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _memory_with_rows(rows: list[dict]) -> ObservationMemory:
    """ObservationMemory backed by the PostgreSQL test DB, seeded with rows.

    Each call uses a unique person_id so rows from different calls are isolated.
    """
    person_id = str(uuid.uuid4())
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[1.0, 0.0, 0.0]]),
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]),
    ):
        mem = ObservationMemory(person_id=person_id)

    now_str = datetime.now().isoformat()
    with mem._db_lock:
        conn = mem._ensure_connected()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO persons (id,name,display_name,created_at,updated_at) "
                "VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (person_id, f"test-{person_id[:8]}", "Test Person", now_str, now_str),
            )
        for row in rows:
            obs_ts = datetime.strptime(row["date"], "%Y-%m-%d").replace(hour=12, minute=0)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO observations "
                    "(id,content,timestamp,direction,kind,emotion) "
                    "VALUES (%s,%s,%s,%s,%s,%s)",
                    (
                        str(uuid.uuid4()),
                        row["content"],
                        obs_ts,
                        "unknown",
                        row.get("kind", "conversation"),
                        row.get("emotion", "neutral"),
                    ),
                )
        conn.commit()

    return mem


def _today_str() -> str:
    return date.today().strftime("%Y-%m-%d")


def _last_year_same_day() -> str:
    today = date.today()
    try:
        return today.replace(year=today.year - 1).strftime("%Y-%m-%d")
    except ValueError:
        # Feb 29 edge case
        return (today - timedelta(days=365)).strftime("%Y-%m-%d")


def _oif(mem):
    """記憶の広がりは口が答える（環-e-い）。**口は本物・内側の記憶だけ偽物**にする。"""
    from familiar_agent.io.oif import OIF

    return OIF(mem)


def _weeks_ago(n: int) -> str:
    return (date.today() - timedelta(weeks=n)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Tests: recall_on_this_day / recall_on_this_day_async
# ---------------------------------------------------------------------------


def test_recall_on_this_day_returns_matching_month_day() -> None:
    anniversary_date = _last_year_same_day()
    mem = _memory_with_rows(
        [
            {"content": "カメラを設置した日", "date": anniversary_date},
            {"content": "全然関係ない日", "date": _weeks_ago(4)},
        ]
    )
    today = date.today()
    results = mem.recall_on_this_day(today.month, today.day)
    assert len(results) == 1
    assert "カメラを設置した日" in results[0]["content"]


def test_recall_on_this_day_excludes_today() -> None:
    """Today's memories should NOT appear — only past years."""
    mem = _memory_with_rows(
        [
            {"content": "今日の記憶", "date": _today_str()},
        ]
    )
    today = date.today()
    results = mem.recall_on_this_day(today.month, today.day)
    assert len(results) == 0


def test_recall_on_this_day_empty_db_returns_empty() -> None:
    mem = _memory_with_rows([])
    today = date.today()
    results = mem.recall_on_this_day(today.month, today.day)
    assert results == []


def test_recall_on_this_day_respects_n_limit() -> None:
    anniversary_date = _last_year_same_day()
    mem = _memory_with_rows([{"content": f"記憶{i}", "date": anniversary_date} for i in range(5)])
    today = date.today()
    results = mem.recall_on_this_day(today.month, today.day, n=2)
    assert len(results) <= 2


@pytest.mark.asyncio
async def test_recall_on_this_day_async_works() -> None:
    anniversary_date = _last_year_same_day()
    mem = _memory_with_rows(
        [
            {"content": "去年の今日", "date": anniversary_date},
        ]
    )
    today = date.today()
    results = await mem.recall_on_this_day_async(today.month, today.day)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Tests: get_earliest_date / get_earliest_date_async
# ---------------------------------------------------------------------------


def test_get_earliest_date_returns_min_date() -> None:
    mem = _memory_with_rows(
        [
            {"content": "新しい記憶", "date": _weeks_ago(1)},
            {"content": "古い記憶", "date": _weeks_ago(10)},
        ]
    )
    result = mem.get_earliest_date()
    assert result == _weeks_ago(10)


def test_get_earliest_date_empty_db_returns_none() -> None:
    mem = _memory_with_rows([])
    result = mem.get_earliest_date()
    assert result is None


@pytest.mark.asyncio
async def test_get_earliest_date_async_works() -> None:
    mem = _memory_with_rows(
        [
            {"content": "古い記憶", "date": _weeks_ago(5)},
        ]
    )
    result = await mem.get_earliest_date_async()
    assert result == _weeks_ago(5)


# 起動からの節目を system 文へ添える `_anniversary_context` は、旧 `run()` の残りとして環-ab で外した。
