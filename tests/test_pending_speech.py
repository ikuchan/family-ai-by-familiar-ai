"""Tests for PendingSpeechStore (Issue D).

pending_speech テーブルへの追加・一覧・削除・鮮度計算・失効判定・FK CASCADE を検証する。
observations とは独立し、想起系テーブルには触れない。
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
import pytest
from unittest.mock import patch

from familiar_agent.config import PendingSpeechConfig
from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel
from familiar_agent.tools.pending_speech_store import PendingSpeechStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TEST_DB_URL = os.environ.get(
    "DATABASE_URL",
    os.environ["DATABASE_URL"],
)


def _pg_conn():
    return psycopg2.connect(_TEST_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def _make_memory() -> ObservationMemory:
    person_id = str(uuid.uuid4())
    now_str = datetime.now().isoformat()
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO persons (id,name,display_name,created_at,updated_at) "
                "VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (person_id, f"test-{person_id[:8]}", "Test", now_str, now_str),
            )
        conn.commit()
    finally:
        conn.close()
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[1.0, 0.0, 0.0]]),
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]),
    ):
        return ObservationMemory(person_id=person_id)


def _insert_obs(memory: ObservationMemory, content: str, superseded_by: str | None = None) -> str:
    obs_id = str(uuid.uuid4())
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations "
                "(id,content,timestamp,direction,kind,emotion,superseded_by) "
                "VALUES (%s,%s,now(),%s,%s,%s,%s)",
                (obs_id, content, "unknown", "conversation", "neutral", superseded_by),
            )
        conn.commit()
    finally:
        conn.close()
    return obs_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store():
    import psycopg2 as _psycopg2
    from familiar_agent.db_migrations import apply_migrations, default_migration_dir
    _mc = _psycopg2.connect(_TEST_DB_URL)
    apply_migrations(_mc, default_migration_dir())
    _mc.commit()
    _mc.close()

    s = PendingSpeechStore(database_url=_TEST_DB_URL)
    yield s
    if s._conn is not None and not s._conn.closed:
        try:
            s._conn.rollback()
            s._conn.close()
        except Exception:
            pass


@pytest.fixture()
def memory():
    return _make_memory()


@pytest.fixture()
def cfg():
    return PendingSpeechConfig(
        half_life_days=1.0,
        floor=0.01,
        expire_threshold=0.1,
        max_per_turn=2,
    )


# ---------------------------------------------------------------------------
# Tests: add
# ---------------------------------------------------------------------------


def test_add_requires_existing_observation(store):
    """実在しない observation_id は拒否(None)。"""
    assert store.add("nonexistent-id", None) is None


def test_add_with_existing_observation(store, memory):
    """実在する observation_id は受理して ID を返す。"""
    obs_id = _insert_obs(memory, "覚えた内容")
    pid = store.add(obs_id, None)
    assert pid is not None
    assert isinstance(pid, str)


def test_add_with_target_person(store, memory):
    """target_person_id を指定して登録できる。"""
    obs_id = _insert_obs(memory, "特定の人に話したい")
    pid = store.add(obs_id, "person-xyz")
    assert pid is not None


# ---------------------------------------------------------------------------
# Tests: list_active
# ---------------------------------------------------------------------------


def test_list_active_returns_added_rows(store, memory):
    obs_id = _insert_obs(memory, "リスト確認")
    store.add(obs_id, None)
    rows = store.list_active()
    obs_ids = [r["observation_id"] for r in rows]
    assert obs_id in obs_ids


def test_list_active_includes_content(store, memory):
    """list_active の各行に observations.content が含まれる。"""
    obs_id = _insert_obs(memory, "コンテンツ確認")
    store.add(obs_id, None)
    rows = store.list_active()
    row = next(r for r in rows if r["observation_id"] == obs_id)
    assert row.get("content") == "コンテンツ確認"


def test_list_active_includes_superseded_by(store, memory):
    """list_active の各行に observations.superseded_by が含まれる。"""
    obs_id = _insert_obs(memory, "superseded確認")
    store.add(obs_id, None)
    rows = store.list_active()
    row = next(r for r in rows if r["observation_id"] == obs_id)
    assert "superseded_by" in row


# ---------------------------------------------------------------------------
# Tests: freshness_score
# ---------------------------------------------------------------------------


def test_freshness_score_near_one_when_fresh(store, memory, cfg):
    """直後は鮮度がほぼ 1.0。"""
    obs_id = _insert_obs(memory, "新鮮な記憶")
    store.add(obs_id, None)
    rows = store.list_active()
    row = next(r for r in rows if r["observation_id"] == obs_id)
    now_epoch = datetime.now(timezone.utc).timestamp()
    score = store.freshness_score(row, now_epoch, cfg)
    assert score > 0.95


def test_freshness_decays(store, memory, cfg):
    """created_at が過去になると鮮度が下がる。"""
    obs_id = _insert_obs(memory, "古い記憶")
    store.add(obs_id, None)

    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE pending_speech SET created_at = now() - interval '2 days' "
                "WHERE observation_id = %s",
                (obs_id,),
            )
        conn.commit()
    finally:
        conn.close()

    rows = store.list_active()
    row = next(r for r in rows if r["observation_id"] == obs_id)
    now_epoch = datetime.now(timezone.utc).timestamp()
    score = store.freshness_score(row, now_epoch, cfg)
    assert score < 1.0


# ---------------------------------------------------------------------------
# Tests: is_expired
# ---------------------------------------------------------------------------


def test_expired_by_threshold(store, memory, cfg):
    """鮮度が expire_threshold を下回ると is_expired=True。"""
    obs_id = _insert_obs(memory, "期限切れ記憶")
    store.add(obs_id, None)

    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE pending_speech SET created_at = now() - interval '100 days' "
                "WHERE observation_id = %s",
                (obs_id,),
            )
        conn.commit()
    finally:
        conn.close()

    rows = store.list_active()
    row = next(r for r in rows if r["observation_id"] == obs_id)
    now_epoch = datetime.now(timezone.utc).timestamp()
    score = store.freshness_score(row, now_epoch, cfg)
    assert store.is_expired(row, score, cfg)


def test_not_expired_when_fresh(store, memory, cfg):
    """直後は期限切れでない。"""
    obs_id = _insert_obs(memory, "新鮮")
    store.add(obs_id, None)
    rows = store.list_active()
    row = next(r for r in rows if r["observation_id"] == obs_id)
    now_epoch = datetime.now(timezone.utc).timestamp()
    score = store.freshness_score(row, now_epoch, cfg)
    assert not store.is_expired(row, score, cfg)


def test_expired_by_supersede(store, memory, cfg):
    """参照先 observation が superseded されたら is_expired=True。"""
    obs_id = _insert_obs(memory, "supersededされる記憶")
    store.add(obs_id, None)

    new_id = _insert_obs(memory, "新しい記憶")
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE observations SET superseded_by = %s WHERE id = %s",
                (new_id, obs_id),
            )
        conn.commit()
    finally:
        conn.close()

    rows = store.list_active()
    row = next((r for r in rows if r["observation_id"] == obs_id), None)
    if row:
        now_epoch = datetime.now(timezone.utc).timestamp()
        score = store.freshness_score(row, now_epoch, cfg)
        assert store.is_expired(row, score, cfg)


# ---------------------------------------------------------------------------
# Tests: delete / cascade
# ---------------------------------------------------------------------------


def test_delete_removes_row(store, memory):
    obs_id = _insert_obs(memory, "削除テスト")
    pid = store.add(obs_id, None)
    store.delete(pid)
    rows = store.list_active()
    ids = [r["id"] for r in rows]
    assert pid not in ids


def test_cascade_delete(store, memory):
    """observation 削除で pending も消える (FK CASCADE)。"""
    obs_id = _insert_obs(memory, "CASCADEテスト")
    pid = store.add(obs_id, None)
    assert pid is not None

    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM observations WHERE id = %s", (obs_id,))
        conn.commit()
    finally:
        conn.close()

    rows = store.list_active()
    ids = [r["id"] for r in rows]
    assert pid not in ids
