"""顔と声の特徴の置き場（知-ae 段 1・2026-10-02・本人の決定ア）。

顔と声の特徴を PostgreSQL の 1 つの表（`recognition_embeddings`・073）に持つ。以前は pickle ファイル
（`EmbeddingStore`）で、開発ルール「保存は PostgreSQL のみ」に反していた。

- 種類（`face`・`voice`）と出どころ（`registered`＝登録の声・`today`＝今日の声）を列で分ける。
- 照らすときは人ごとの重心（平均）を使う。1 つずつ比べて最も近いものを取ると、たまたま似た 1 つに引かれる。
- 上限を超えたら古いものから捨てる（登録の声 30・今日の声 10・本人の決定ア・仮の値）。
- 今日の声は日付が変わると引かない（前の日の分は捨てる）。
- 人を消すと特徴も消える。
"""

from __future__ import annotations

import os
import uuid
from datetime import date, datetime, timedelta, timezone

import numpy as np
import psycopg2
import psycopg2.extras
import pytest

from familiar_agent.store.recognition_embeddings import RecognitionEmbeddingStore

TODAY = date(2026, 10, 2)
NOW = datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def conn():
    from familiar_agent.db import get_db

    get_db()  # マイグレーションを当てておく
    c = psycopg2.connect(os.environ["DATABASE_URL"], cursor_factory=psycopg2.extras.RealDictCursor)
    c.autocommit = True
    yield c
    c.close()


def _person(conn, name: str) -> str:
    pid = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO persons (id, name, display_name, created_at, updated_at) VALUES (%s,%s,'',%s,%s)",
            (pid, name, NOW.isoformat(), NOW.isoformat()),
        )
    return pid


def _vec(*xs: float) -> np.ndarray:
    return np.asarray(xs, dtype=np.float32)


def test_the_table_exists(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('recognition_embeddings') AS t")
        assert cur.fetchone()["t"] == "recognition_embeddings"


def test_the_centroid_is_the_mean_per_person(conn):
    s = RecognitionEmbeddingStore(conn)
    papa, taiki = _person(conn, "雄輔"), _person(conn, "泰輝")
    s.add(papa, "voice", "registered", _vec(1, 0), cap=30, now=NOW)
    s.add(papa, "voice", "registered", _vec(0, 1), cap=30, now=NOW)
    s.add(taiki, "voice", "registered", _vec(1, 1), cap=30, now=NOW)
    got = s.centroids("voice", "registered")
    assert set(got) == {papa, taiki}
    assert np.allclose(got[papa] / np.linalg.norm(got[papa]), _vec(1, 1) / np.sqrt(2))
    assert s.centroids("face", "registered") == {}


def test_over_the_cap_the_oldest_go(conn):
    s = RecognitionEmbeddingStore(conn)
    papa = _person(conn, "雄輔")
    for i in range(4):
        s.add(
            papa,
            "voice",
            "today",
            _vec(float(i), 1),
            cap=3,
            day=TODAY,
            now=NOW + timedelta(seconds=i),
        )
    assert s.count(papa, "voice", "today") == 3
    c = s.centroids("voice", "today", day=TODAY)[papa]
    kept = [_vec(float(i), 1) / np.linalg.norm(_vec(float(i), 1)) for i in (1, 2, 3)]
    assert np.allclose(c, np.mean(kept, axis=0))  # 1・2・3 が残る（0 が捨てられた）


def test_today_is_only_today(conn):
    s = RecognitionEmbeddingStore(conn)
    papa = _person(conn, "雄輔")
    s.add(papa, "voice", "today", _vec(1, 0), cap=10, day=TODAY - timedelta(days=1), now=NOW)
    assert s.centroids("voice", "today", day=TODAY) == {}
    assert s.drop_old_today(TODAY) == 1
    assert s.count(papa, "voice", "today") == 0


def test_deleting_a_person_deletes_the_features(conn):
    s = RecognitionEmbeddingStore(conn)
    papa = _person(conn, "雄輔")
    s.add(papa, "face", "registered", _vec(1, 0), cap=30, now=NOW)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM persons WHERE id = %s", (papa,))
    assert s.centroids("face", "registered") == {}


def test_a_bad_kind_or_origin_is_refused(conn):
    s = RecognitionEmbeddingStore(conn)
    papa = _person(conn, "雄輔")
    with pytest.raises(ValueError):
        s.add(papa, "smell", "registered", _vec(1, 0), cap=30)
    with pytest.raises(ValueError):
        s.add(papa, "voice", "guess", _vec(1, 0), cap=30)
