"""記-a-ろ-ろ：$I$ の計測と $n$ の減り（`出来事を畳む` §4 の 1・2・2026-09-15）。

毎晩、核（$n \ge 1$ の面）の使われる情報量 $I=\sum b_i u_i$ を測って計測ログに残し、
$I>I^\*$ なら**前回の内省以降に参照されなかった**核の $n$ を $\Delta$ 減らす（1 未満にしない）。
「参照された時」は面の列 `last_referred_at`（062）に持ち、`apply_verdicts` が
`important`／`referred` のときに記す。
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import psycopg2
import psycopg2.extras

from familiar_agent.person_memory_manager import AGENT_SELF_ID

_DB_URL = os.environ["DATABASE_URL"]
_VEC = "[" + ",".join(["0.01"] * 1024) + "]"


def _conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _plant(cur, obs_id: str, content: str, ts: datetime, *, direction: str = "発話") -> None:
    cur.execute(
        "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, "
        " groundedness_g0) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (obs_id, content, ts, direction, "observation", "neutral", 0.5),
    )


def _facet(cur, obs_id: str, person_id: str, *, n: int = 0, referred_at=None) -> None:
    cur.execute(
        "INSERT INTO situated_memories (id, obs_id, person_id, vector, relation_key, "
        " groundedness_n, last_referred_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (str(uuid.uuid4()), obs_id, person_id, _VEC, "present", n, referred_at),
    )


def _memory():
    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory().for_person(AGENT_SELF_ID)


def _facet_row(obs_id: str, person_id: str = AGENT_SELF_ID) -> dict:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT groundedness_n, last_referred_at FROM situated_memories "
                "WHERE obs_id=%s AND person_id=%s",
                (obs_id, person_id),
            )
            return dict(cur.fetchone())
    finally:
        conn.close()


# ---- 062：参照された時の列 -----------------------------------------------------


def test_the_facet_has_a_last_referred_at_column() -> None:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT data_type, is_nullable FROM information_schema.columns "
            "WHERE table_name='situated_memories' AND column_name='last_referred_at'"
        )
        row = cur.fetchone()
    assert row and row["data_type"] == "timestamp with time zone" and row["is_nullable"] == "YES"


def test_important_and_referred_stamp_the_time_but_useless_and_unused_do_not() -> None:
    ids = [str(uuid.uuid4()) for _ in range(4)]
    conn = _conn()
    try:
        with conn.cursor() as cur:
            for i in ids:
                _plant(cur, i, f"参照の印_{i}", datetime.now(timezone.utc))
                _facet(cur, i, AGENT_SELF_ID)
    finally:
        conn.close()
    mem = _memory()
    mem._observations.apply_verdicts(
        {ids[0]: "important", ids[1]: "referred", ids[2]: "useless", ids[3]: "unused"}
    )
    assert _facet_row(ids[0])["last_referred_at"] is not None
    assert _facet_row(ids[1])["last_referred_at"] is not None
    assert _facet_row(ids[2])["last_referred_at"] is None
    assert _facet_row(ids[3])["last_referred_at"] is None


# ---- 店：核の面・今日の分・減り ------------------------------------------------


def _plant_rest(cur, ts: datetime) -> None:
    _plant(cur, str(uuid.uuid4()), "内省を回した（…）", ts, direction="内省")


def test_core_faces_lists_every_face_with_n_at_least_one_across_viewpoints() -> None:
    a, b, c = (str(uuid.uuid4()) for _ in range(3))
    other = str(uuid.uuid4())
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO persons (id, name, display_name, created_at, updated_at) "
                "VALUES (%s, %s, %s, now()::text, now()::text)",
                (other, f"人_{other[:8]}", "人"),
            )
            now = datetime.now(timezone.utc)
            _plant(cur, a, "核の記録 あ", now - timedelta(days=3))
            _plant(cur, b, "核の記録 い", now - timedelta(days=1))
            _plant(cur, c, "核でない記録", now)
            _facet(cur, a, AGENT_SELF_ID, n=2)
            _facet(cur, a, other, n=1)  # 同じ出来事でも面が違えば別に数える
            _facet(cur, b, AGENT_SELF_ID, n=1)
            _facet(cur, c, AGENT_SELF_ID, n=0)
    finally:
        conn.close()
    rows = _memory()._observations.core_faces()
    got = {(r["obs_id"], r["person_id"]): r for r in rows}
    assert set(got) == {(a, AGENT_SELF_ID), (a, other), (b, AGENT_SELF_ID)}
    r = got[(a, AGENT_SELF_ID)]
    assert r["groundedness_n"] == 2 and r["chars"] == len("核の記録 あ")
    assert r["groundedness_g0"] == 0.5 and r["timestamp"] is not None


def test_fresh_since_last_rest_returns_what_came_after_the_last_rest() -> None:
    old, new = str(uuid.uuid4()), str(uuid.uuid4())
    conn = _conn()
    try:
        with conn.cursor() as cur:
            now = datetime.now(timezone.utc)
            _plant(cur, old, "前の晩より前", now - timedelta(hours=30))
            _plant_rest(cur, now - timedelta(hours=20))
            _plant(cur, new, "前の晩より後", now - timedelta(hours=2))
    finally:
        conn.close()
    rows = _memory()._observations.fresh_since_last_rest()
    ids = {r["obs_id"] for r in rows}
    assert new in ids and old not in ids
    assert all("chars" in r and "timestamp" in r for r in rows)


def test_decay_lowers_unreferenced_cores_by_delta_but_never_below_one() -> None:
    now = datetime.now(timezone.utc)
    stale3, stale1, fresh, noncore = (str(uuid.uuid4()) for _ in range(4))
    conn = _conn()
    try:
        with conn.cursor() as cur:
            _plant_rest(cur, now - timedelta(hours=20))
            for i in (stale3, stale1, fresh, noncore):
                _plant(cur, i, f"減りの対象_{i}", now - timedelta(days=2))
            _facet(cur, stale3, AGENT_SELF_ID, n=3, referred_at=now - timedelta(days=5))
            _facet(cur, stale1, AGENT_SELF_ID, n=1, referred_at=None)
            _facet(cur, fresh, AGENT_SELF_ID, n=3, referred_at=now - timedelta(hours=1))
            _facet(cur, noncore, AGENT_SELF_ID, n=0, referred_at=None)
    finally:
        conn.close()
    moved = _memory()._observations.decay_groundedness(2)
    assert moved == 1  # stale3 だけ動く（stale1 は 1 のまま・fresh は参照済み・noncore は核でない）
    assert _facet_row(stale3)["groundedness_n"] == 1
    assert _facet_row(stale1)["groundedness_n"] == 1
    assert _facet_row(fresh)["groundedness_n"] == 3
    assert _facet_row(noncore)["groundedness_n"] == 0


def test_decay_of_zero_touches_nothing() -> None:
    i = str(uuid.uuid4())
    conn = _conn()
    try:
        with conn.cursor() as cur:
            _plant(cur, i, "動かない", datetime.now(timezone.utc))
            _facet(cur, i, AGENT_SELF_ID, n=4)
    finally:
        conn.close()
    assert _memory()._observations.decay_groundedness(0) == 0
    assert _facet_row(i)["groundedness_n"] == 4
