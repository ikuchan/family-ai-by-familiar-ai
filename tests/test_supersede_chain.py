"""Tests for the supersede-chain read helper (_read_supersede_chain).

系統B 畳み込みの読み出し側（§7）。現行版 MI（superseded_by IS NULL）を起点に
superseded_by の祖先を再帰でたどり、信念の改訂履歴（版チェーン）を再構成する
dumb な読み出し。採点・想起判断は持たず、既存経路からは未接続。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import psycopg2

from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel
from familiar_agent.person_memory_manager import AGENT_SELF_ID


_DB_URL = "postgresql://familiar:familiar@localhost:5433/familiar_test"
_NOW = datetime(2026, 6, 1, 12, 0, 0)


def _insert_obs(cur, obs_id, content, ts, superseded_by=None):
    cur.execute(
        "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, "
        "person_id, superseded_by) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
        (obs_id, content, ts, "unknown", "self_model", "neutral", AGENT_SELF_ID, superseded_by),
    )


def _mem():
    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory()


def test_chain_returns_head_then_ancestors() -> None:
    """A←B←C（C が現行版）。head=C から現行版＋祖先を depth 順（新→旧）で返す。"""
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        # 現行版 C を先に入れ、B は superseded_by=C、A は superseded_by=B
        _insert_obs(cur, "v-c", "third", _NOW, superseded_by=None)
        _insert_obs(cur, "v-b", "second", _NOW - timedelta(hours=1), superseded_by="v-c")
        _insert_obs(cur, "v-a", "first", _NOW - timedelta(hours=2), superseded_by="v-b")
    conn.close()

    mem = _mem()
    rows = mem._read_supersede_chain("v-c", ("id", "content"))
    assert [r["content"] for r in rows] == ["third", "second", "first"]


def test_chain_single_live_mi_returns_itself() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "solo-1", "only", _NOW, superseded_by=None)
    conn.close()

    mem = _mem()
    rows = mem._read_supersede_chain("solo-1", ("id", "content"))
    assert [r["content"] for r in rows] == ["only"]


def test_chain_gathers_merged_ancestors() -> None:
    """多対一収束（重複を最古でなく現行版へ畳んだ形＝2つの旧が同じ新を指す）でも祖先を集める。"""
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "m-head", "head", _NOW, superseded_by=None)
        _insert_obs(cur, "m-x", "merged-x", _NOW - timedelta(hours=1), superseded_by="m-head")
        _insert_obs(cur, "m-y", "merged-y", _NOW - timedelta(hours=2), superseded_by="m-head")
    conn.close()

    mem = _mem()
    rows = mem._read_supersede_chain("m-head", ("id", "content"))
    assert {r["content"] for r in rows} == {"head", "merged-x", "merged-y"}


def test_chain_returns_empty_for_unknown_head() -> None:
    mem = _mem()
    rows = mem._read_supersede_chain("no-such-id", ("id", "content"))
    assert rows == []
