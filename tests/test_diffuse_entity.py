"""拡散想起スライス2 (B) エンティティ辺：視点列から種 person を選び person 中心で再想起。"""

from __future__ import annotations

import os
import uuid

import psycopg2

from familiar_agent.core.diffuse import select_entity_seeds
from familiar_agent.diffuse_store import recall_by_person


# ── select_entity_seeds（純関数・subject 優先・除外・重複除去） ───────────────

def test_seeds_subject_first_then_participants_then_writer():
    persp = [
        {"subject_id": "PA", "participants": ["PA", "PB"], "writer_id": "AGENT"},
        {"subject_id": "PC", "participants": ["PC"], "writer_id": "PA"},
    ]
    out = select_entity_seeds(persp, exclude={"AGENT", "SPEAKER"})
    assert out == ["PA", "PC", "PB"]  # subject PA,PC → participant PB（新規）→ writer は既出/除外


def test_seeds_excludes_and_dedups_and_skips_none():
    persp = [{"subject_id": None, "participants": ["X", None, "X"], "writer_id": "SELF"}]
    assert select_entity_seeds(persp, exclude={"SELF"}) == ["X"]


def test_seeds_empty():
    assert select_entity_seeds([], set()) == []


# ── recall_by_person（実 DB・subject/participant 一致・新しい順・現行版のみ） ──

def _conn():
    c = psycopg2.connect(os.environ["DATABASE_URL"])
    c.autocommit = True
    return c


def _insert_obs(cur, *, subject=None, participants="[]", supseded=False, ts="2020-01-01"):
    oid = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, "
        "participants_json, subject_id, superseded_by) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (oid, "c", ts, "会話", "conversation", "neutral",
         participants, subject, (str(uuid.uuid4()) if supseded else None)),
    )
    return oid


def test_recall_by_person_matches_subject_and_participant_recent_first():
    conn = _conn()
    tag = uuid.uuid4().hex[:8]
    X = f"PERSON-{tag}"
    with conn.cursor() as cur:
        old = _insert_obs(cur, subject=X, ts="2020-01-01")          # subject 一致・古い
        new = _insert_obs(cur, participants=f'["{X}"]', ts="2025-01-01")  # participant 一致・新しい
        _insert_obs(cur, subject=X, supseded=True)                   # 旧版は除外
        _insert_obs(cur, subject="OTHER")                            # 無関係
    got = recall_by_person(conn, X, limit=5)
    conn.close()
    assert set(got) == {old, new}          # subject と participant の両方を拾う
    assert got[0] == new                    # 新しい順（participant の新しい方が先頭）


# ── diffuse_ids（有界再帰・純関数・候補取得はコールバック注入） ───────────────

from familiar_agent.core.diffuse import diffuse_ids  # noqa: E402


def _grapher(graph):
    def get(known):
        out = []
        for k in known:
            out += graph.get(k, [])
        return out
    return get


def test_diffuse_recurses_until_dry():
    # a→x, x→y, y→(なし)。深く再帰して x,y を足す
    get = _grapher({"a": ["x"], "x": ["y"]})
    assert diffuse_ids(["a"], get, max_add=5, max_depth=3) == ["x", "y"]


def test_diffuse_respects_max_add():
    get = _grapher({"a": ["x"], "x": ["y"]})
    assert diffuse_ids(["a"], get, max_add=1, max_depth=3) == ["x"]


def test_diffuse_respects_max_depth():
    get = _grapher({"a": ["x"], "x": ["y"]})
    assert diffuse_ids(["a"], get, max_add=5, max_depth=1) == ["x"]


def test_diffuse_excludes_seed_and_dedups():
    get = _grapher({"a": ["a", "x", "x"]})  # 自分・重複を返しても
    assert diffuse_ids(["a"], get, max_add=5, max_depth=2) == ["x"]


def test_diffuse_empty_when_no_candidates():
    assert diffuse_ids(["a"], _grapher({}), max_add=5, max_depth=3) == []
