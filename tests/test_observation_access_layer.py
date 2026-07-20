"""Tests for the thin observation access layer (_read_observations_by_kind)."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import psycopg2
import psycopg2.extras

from familiar_agent.store import clock
from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel
from familiar_agent.person_memory_manager import AGENT_SELF_ID, DEFAULT_PERSON_ID


_DB_URL = "postgresql://familiar:familiar@localhost:5433/familiar_test"

# ローカルの生活時間で「その日の正午」を表す tz 付き instant。
# naive のままだと挿入セッションの TZ に意味が左右され（DB セッション TZ を
# ローカルへ固定した後は表示がずれる）、date/time の期待値が壊れる。
_NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=clock.local_tz())


def _insert_obs(cur, obs_id: str, content: str, kind: str, person_id: str, ts: datetime) -> None:
    cur.execute(
        "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, person_id) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (obs_id, content, ts, "unknown", kind, "neutral", person_id),
    )


def _mem() -> ObservationMemory:
    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory()


# ── 1. _read_observations_by_kind: 順序と件数制限 ────────────────────────────

def test_read_observations_by_kind_returns_newest_first() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "old-1", "old curiosity", "curiosity", AGENT_SELF_ID, _NOW - timedelta(hours=2))
        _insert_obs(cur, "mid-2", "mid curiosity", "curiosity", AGENT_SELF_ID, _NOW - timedelta(hours=1))
        _insert_obs(cur, "new-3", "new curiosity", "curiosity", AGENT_SELF_ID, _NOW)
    conn.close()

    mem = _mem()
    rows = mem._observations._read_observations_by_kind("curiosity", AGENT_SELF_ID, 3, ("content", "timestamp"))

    assert len(rows) == 3
    assert rows[0]["content"] == "new curiosity"
    assert rows[1]["content"] == "mid curiosity"
    assert rows[2]["content"] == "old curiosity"


def test_read_observations_by_kind_respects_limit() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        for i in range(5):
            _insert_obs(cur, f"c-{i}", f"curiosity {i}", "curiosity", AGENT_SELF_ID,
                        _NOW + timedelta(minutes=i))
    conn.close()

    mem = _mem()
    rows = mem._observations._read_observations_by_kind("curiosity", AGENT_SELF_ID, 3, ("content", "timestamp"))

    assert len(rows) == 3


# ── 2. kind と person_id でフィルタされること ──────────────────────────────

def test_read_observations_by_kind_filters_by_kind() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "obs-1", "curiosity row", "curiosity", AGENT_SELF_ID, _NOW)
        _insert_obs(cur, "obs-2", "observation row", "observation", AGENT_SELF_ID, _NOW + timedelta(seconds=1))
        _insert_obs(cur, "obs-3", "feeling row", "feeling", AGENT_SELF_ID, _NOW + timedelta(seconds=2))
    conn.close()

    mem = _mem()
    rows = mem._observations._read_observations_by_kind("curiosity", AGENT_SELF_ID, 10, ("content", "timestamp"))

    assert len(rows) == 1
    assert rows[0]["content"] == "curiosity row"


def test_read_observations_by_kind_filters_by_person_id() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "self-1", "agent curiosity", "curiosity", AGENT_SELF_ID, _NOW)
        _insert_obs(cur, "user-1", "user curiosity", "curiosity", DEFAULT_PERSON_ID, _NOW + timedelta(seconds=1))
    conn.close()

    mem = _mem()
    rows = mem._observations._read_observations_by_kind("curiosity", AGENT_SELF_ID, 10, ("content", "timestamp"))

    assert len(rows) == 1
    assert rows[0]["content"] == "agent curiosity"


def test_read_observations_by_kind_returns_empty_when_none_match() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "obs-1", "some feeling", "feeling", AGENT_SELF_ID, _NOW)
    conn.close()

    mem = _mem()
    rows = mem._observations._read_observations_by_kind("curiosity", AGENT_SELF_ID, 10, ("content", "timestamp"))

    assert rows == []


# ── 3. recall_curiosities の付け替え後の戻り値の形 ──────────────────────────

def test_recall_curiosities_returns_expected_shape() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "cur-1", "first curiosity", "curiosity", AGENT_SELF_ID, _NOW - timedelta(hours=1))
        _insert_obs(cur, "cur-2", "second curiosity", "curiosity", AGENT_SELF_ID, _NOW)
    conn.close()

    mem = _mem()
    result = mem.recall_curiosities(n=5)

    assert len(result) == 2
    newest = result[0]
    assert set(newest.keys()) == {"summary", "date", "time"}
    assert newest["summary"] == "second curiosity"
    assert newest["date"] == "2026-06-01"
    assert newest["time"] == "12:00"


def test_recall_curiosities_returns_empty_when_none() -> None:
    mem = _mem()
    result = mem.recall_curiosities(n=5)
    assert result == []


# ── 4. recall_self_model の付け替え後の戻り値の形（emotion 込み経路） ────────

def _insert_obs_with_emotion(
    cur, obs_id: str, content: str, kind: str, person_id: str, ts: datetime, emotion: str
) -> None:
    cur.execute(
        "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, person_id) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (obs_id, content, ts, "unknown", kind, emotion, person_id),
    )


# situated_embeddings.vector は vector(1024)。recall_day_summaries は situated 相関で
# 引くため、返ってほしい観測には対象 person の situated 行が要る。ベクトルは順序に
# 使わない（timestamp DESC）が、コサイン索引がゼロノルムを嫌うので非ゼロを入れる。
_VEC = "[" + ",".join(["1"] + ["0"] * 1023) + "]"


def _insert_situated(cur, se_id: str, obs_id: str, person_id: str) -> None:
    cur.execute(
        "INSERT INTO situated_embeddings (id, obs_id, person_id, vector) VALUES (%s, %s, %s, %s)",
        (se_id, obs_id, person_id, _VEC),
    )


def test_recall_self_model_returns_expected_shape_newest_first_with_limit() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs_with_emotion(cur, "sm-1", "first self model", "self_model", AGENT_SELF_ID,
                                  _NOW - timedelta(hours=2), "neutral")
        _insert_obs_with_emotion(cur, "sm-2", "second self model", "self_model", AGENT_SELF_ID,
                                  _NOW - timedelta(hours=1), "neutral")
        _insert_obs_with_emotion(cur, "sm-3", "third self model", "self_model", AGENT_SELF_ID,
                                  _NOW, "neutral")
    conn.close()

    mem = _mem()
    result = mem.recall_self_model(n=2)

    assert len(result) == 2
    newest = result[0]
    assert set(newest.keys()) == {"summary", "date", "time", "emotion"}
    assert newest["summary"] == "third self model"
    assert newest["date"] == "2026-06-01"
    assert newest["time"] == "12:00"
    assert result[1]["summary"] == "second self model"


def test_recall_self_model_returns_distinct_emotion_values_unchanged() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs_with_emotion(cur, "sm-emo-1", "neutral self model", "self_model", AGENT_SELF_ID,
                                  _NOW - timedelta(hours=1), "neutral")
        _insert_obs_with_emotion(cur, "sm-emo-2", "happy self model", "self_model", AGENT_SELF_ID,
                                  _NOW, "happy")
    conn.close()

    mem = _mem()
    result = mem.recall_self_model(n=5)

    by_summary = {r["summary"]: r["emotion"] for r in result}
    assert by_summary["happy self model"] == "happy"
    assert by_summary["neutral self model"] == "neutral"


def test_recall_self_model_filters_by_agent_self_id() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs_with_emotion(cur, "sm-self", "agent self model", "self_model", AGENT_SELF_ID,
                                  _NOW, "neutral")
        _insert_obs_with_emotion(cur, "sm-other", "other person self model", "self_model",
                                  DEFAULT_PERSON_ID, _NOW + timedelta(seconds=1), "neutral")
    conn.close()

    mem = _mem()
    result = mem.recall_self_model(n=10)

    assert len(result) == 1
    assert result[0]["summary"] == "agent self model"


def test_recall_self_model_returns_empty_when_none() -> None:
    mem = _mem()
    result = mem.recall_self_model(n=5)
    assert result == []


# ── 5. recall_day_summaries の付け替え後の戻り値の形（situated 相関経路） ──

def test_recall_day_summaries_returns_expected_shape_newest_first_with_limit() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs_with_emotion(cur, "ds-1", "first day summary", "day_summary", DEFAULT_PERSON_ID,
                                  _NOW - timedelta(hours=2), "neutral")
        _insert_obs_with_emotion(cur, "ds-2", "second day summary", "day_summary", DEFAULT_PERSON_ID,
                                  _NOW - timedelta(hours=1), "neutral")
        _insert_obs_with_emotion(cur, "ds-3", "third day summary", "day_summary", DEFAULT_PERSON_ID,
                                  _NOW, "neutral")
        _insert_situated(cur, "se-ds-1", "ds-1", DEFAULT_PERSON_ID)
        _insert_situated(cur, "se-ds-2", "ds-2", DEFAULT_PERSON_ID)
        _insert_situated(cur, "se-ds-3", "ds-3", DEFAULT_PERSON_ID)
    conn.close()

    mem = _mem()
    assert mem._person_id == DEFAULT_PERSON_ID
    result = mem.recall_day_summaries(n=2)

    assert len(result) == 2
    newest = result[0]
    assert set(newest.keys()) == {"summary", "date", "time", "emotion"}
    assert newest["summary"] == "third day summary"
    assert newest["date"] == "2026-06-01"
    assert newest["time"] == "12:00"
    assert result[1]["summary"] == "second day summary"


def test_recall_day_summaries_correlates_by_situated_not_owner() -> None:
    """付け替え後は所有者絞りでなく situated 相関で引く。所有者が別人でも、この memory の
    person_id の situated 行があれば返り（相関で含む）、situated 行が無ければ所有していても
    返らない（相関が母集合を決める）。"""
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        # 所有者は AGENT_SELF_ID だが DEFAULT_PERSON_ID の situated 行がある → 返る
        _insert_obs_with_emotion(cur, "ds-corr", "correlated day summary", "day_summary",
                                  AGENT_SELF_ID, _NOW, "neutral")
        _insert_situated(cur, "se-corr", "ds-corr", DEFAULT_PERSON_ID)
        # 所有者は DEFAULT_PERSON_ID だが DEFAULT_PERSON_ID の situated 行が無い → 返らない
        _insert_obs_with_emotion(cur, "ds-uncorr", "uncorrelated day summary", "day_summary",
                                  DEFAULT_PERSON_ID, _NOW + timedelta(seconds=1), "neutral")
        _insert_situated(cur, "se-uncorr", "ds-uncorr", AGENT_SELF_ID)
    conn.close()

    mem = _mem()
    assert mem._person_id == DEFAULT_PERSON_ID
    result = mem.recall_day_summaries(n=10)

    assert [r["summary"] for r in result] == ["correlated day summary"]


def test_recall_day_summaries_returns_distinct_emotion_values_unchanged() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs_with_emotion(cur, "ds-emo-1", "neutral day summary", "day_summary",
                                  DEFAULT_PERSON_ID, _NOW - timedelta(hours=1), "neutral")
        _insert_obs_with_emotion(cur, "ds-emo-2", "happy day summary", "day_summary",
                                  DEFAULT_PERSON_ID, _NOW, "happy")
        _insert_situated(cur, "se-emo-1", "ds-emo-1", DEFAULT_PERSON_ID)
        _insert_situated(cur, "se-emo-2", "ds-emo-2", DEFAULT_PERSON_ID)
    conn.close()

    mem = _mem()
    result = mem.recall_day_summaries(n=5)

    by_summary = {r["summary"]: r["emotion"] for r in result}
    assert by_summary["happy day summary"] == "happy"
    assert by_summary["neutral day summary"] == "neutral"


def test_recall_day_summaries_returns_empty_when_none() -> None:
    mem = _mem()
    result = mem.recall_day_summaries(n=5)
    assert result == []


# ── 6. _read_observations_by_kind の複数 kind 対応（tuple） ─────────────────

def test_read_observations_by_kind_accepts_tuple_of_kinds() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "mk-1", "a feeling", "feeling", AGENT_SELF_ID, _NOW - timedelta(hours=2))
        _insert_obs(cur, "mk-2", "a conversation", "conversation", AGENT_SELF_ID, _NOW - timedelta(hours=1))
        _insert_obs(cur, "mk-3", "a curiosity", "curiosity", AGENT_SELF_ID, _NOW)
    conn.close()

    mem = _mem()
    rows = mem._observations._read_observations_by_kind(("feeling", "conversation"), AGENT_SELF_ID, 10, ("content", "timestamp"))

    contents = {r["content"] for r in rows}
    assert contents == {"a feeling", "a conversation"}
    assert "a curiosity" not in contents


def test_read_observations_by_kind_tuple_respects_order_and_limit() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "mk-old", "old feeling", "feeling", AGENT_SELF_ID, _NOW - timedelta(hours=2))
        _insert_obs(cur, "mk-mid", "mid conversation", "conversation", AGENT_SELF_ID, _NOW - timedelta(hours=1))
        _insert_obs(cur, "mk-new", "new feeling", "feeling", AGENT_SELF_ID, _NOW)
    conn.close()

    mem = _mem()
    rows = mem._observations._read_observations_by_kind(("feeling", "conversation"), AGENT_SELF_ID, 2, ("content", "timestamp"))

    assert len(rows) == 2
    assert rows[0]["content"] == "new feeling"
    assert rows[1]["content"] == "mid conversation"


def test_read_observations_by_kind_str_path_unchanged_after_tuple_support() -> None:
    """既存の単一 kind (str) 経路が、複数 kind 対応の追加後も壊れていないことの明示的な回帰確認。"""
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs(cur, "sk-1", "curiosity row", "curiosity", AGENT_SELF_ID, _NOW)
        _insert_obs(cur, "sk-2", "feeling row", "feeling", AGENT_SELF_ID, _NOW + timedelta(seconds=1))
    conn.close()

    mem = _mem()
    rows = mem._observations._read_observations_by_kind("curiosity", AGENT_SELF_ID, 10, ("content", "timestamp"))

    assert len(rows) == 1
    assert rows[0]["content"] == "curiosity row"


# ── 7. recent_feelings の付け替え後の戻り値の形（複数 kind 経路の実証） ─────

def test_recent_feelings_returns_expected_shape_newest_first_with_limit() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs_with_emotion(cur, "rf-1", "first feeling", "feeling", DEFAULT_PERSON_ID,
                                  _NOW - timedelta(hours=2), "neutral")
        _insert_obs_with_emotion(cur, "rf-2", "first conversation", "conversation", DEFAULT_PERSON_ID,
                                  _NOW - timedelta(hours=1), "neutral")
        _insert_obs_with_emotion(cur, "rf-3", "second feeling", "feeling", DEFAULT_PERSON_ID,
                                  _NOW, "happy")
    conn.close()

    mem = _mem()
    assert mem._person_id == DEFAULT_PERSON_ID
    result = mem.recent_feelings(n=2)

    assert len(result) == 2
    newest = result[0]
    assert set(newest.keys()) == {"summary", "date", "time", "emotion"}
    assert newest["summary"] == "second feeling"
    assert newest["emotion"] == "happy"
    assert newest["date"] == "2026-06-01"
    assert newest["time"] == "12:00"
    assert result[1]["summary"] == "first conversation"


def test_recent_feelings_excludes_other_kinds() -> None:
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        _insert_obs_with_emotion(cur, "rf-feeling", "a feeling", "feeling", DEFAULT_PERSON_ID,
                                  _NOW - timedelta(seconds=1), "neutral")
        _insert_obs_with_emotion(cur, "rf-conv", "a conversation", "conversation", DEFAULT_PERSON_ID,
                                  _NOW, "neutral")
        _insert_obs_with_emotion(cur, "rf-curiosity", "a curiosity", "curiosity", DEFAULT_PERSON_ID,
                                  _NOW + timedelta(seconds=1), "neutral")
    conn.close()

    mem = _mem()
    result = mem.recent_feelings(n=10)

    summaries = {r["summary"] for r in result}
    assert summaries == {"a feeling", "a conversation"}


def test_recent_feelings_returns_empty_when_none() -> None:
    mem = _mem()
    result = mem.recent_feelings(n=5)
    assert result == []
