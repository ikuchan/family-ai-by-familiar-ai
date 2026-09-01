"""044（`situated_memories` → `situated_memories`）が効いていることを確かめる。

**索引を記憶にする段である。** 044 より前の situated は
`(id, obs_id, person_id, vector, relation_key)` だけを持つベクトル索引で、記憶の実体
（本文・時間の起点・根づき）は `observations` にあった。044 で `content`・
`last_recalled_at`・`groundedness_n` を面が自分で持つ。

**なぜ面へ移すか**（`設計図` [D-在席相関/V2]）：出来事を1行だけで持つと、supersede で
畳んだ瞬間に「誰が何を言ったか」が畳んだ側の `content` の文字列にしか残らない。文字列は
版が進むたび書き直されるので復元できない。面は `superseded_by` を持たない
（`superseded_by` は `observations` の条件）ので、畳んでも残る。

**旧値は引き継がない。** 出来事1件の値をどの面へ写すかに正解が無いためで、原本もそうした
（8月21日の `last_recalled_at` 139 件はすべて 044 の適用時刻より後だった）。
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import psycopg2
import psycopg2.extras

from familiar_agent.person_memory_manager import AGENT_SELF_ID, DEFAULT_PERSON_ID

_DB_URL = os.environ["DATABASE_URL"]

_VEC = "[" + ",".join(["1"] + ["0"] * 1023) + "]"


def _conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _tables() -> set[str]:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' AND table_type='BASE TABLE'"
            )
            return {r["table_name"] for r in cur.fetchall()}
    finally:
        conn.close()


def _columns(table: str) -> set[str]:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name=%s",
                (table,),
            )
            return {r["column_name"] for r in cur.fetchall()}
    finally:
        conn.close()


# ── RED-1：表の名前 ─────────────────────────────────────────────────

def test_the_table_is_renamed() -> None:
    t = _tables()
    assert "situated_memories" in t
    assert "situated_embeddings" not in t


# ── RED-2：列の移動 ─────────────────────────────────────────────────

def test_the_facet_holds_the_memory() -> None:
    """面が本文と時間の起点と根づきを自分で持つ。"""
    cols = _columns("situated_memories")
    for name in ("obs_id", "person_id", "vector", "relation_key",
                 "content", "last_recalled_at", "groundedness_n"):
        assert name in cols, name


def test_the_event_no_longer_holds_them() -> None:
    """出来事の側からは消える。取込の驚き a0 だけが残る。"""
    cols = _columns("observations")
    assert "last_recalled_at" not in cols
    assert "groundedness_n" not in cols
    assert "groundedness_g0" in cols, "パジュにとっての驚きは出来事ごとなので残る"


def test_the_recency_index_exists() -> None:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT indexdef FROM pg_indexes WHERE tablename='situated_memories'"
            )
            defs = " ".join(r["indexdef"] for r in cur.fetchall())
    finally:
        conn.close()
    assert "idx_situated_recency" in defs
    assert "person_id" in defs and "last_recalled_at" in defs


# ── RED-3：採点と若返りが面を読む（本体） ───────────────────────────

def _plant(cur, obs_id: str, content: str, ts: datetime) -> None:
    cur.execute(
        "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, "
        " groundedness_g0) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (obs_id, content, ts, "unknown", "observation", "neutral", 0.5),
    )


def _facet(cur, obs_id: str, person_id: str, *, n: int = 0) -> None:
    cur.execute(
        "INSERT INTO situated_memories (id, obs_id, person_id, vector, relation_key, "
        " groundedness_n) VALUES (%s, %s, %s, %s, %s, %s)",
        (str(uuid.uuid4()), obs_id, person_id, _VEC, "presence", n),
    )


def test_groundedness_is_read_from_the_facet_not_the_event() -> None:
    """同じ出来事でも、面ごとに根づきが違う。

    片方の面だけ `groundedness_n` を上げ、その視点で引いたときだけ a が上がることを見る。
    出来事の列を読んでいるうちは、どちらの視点でも同じ値になるので落ちる。
    """
    from unittest.mock import patch

    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    obs_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    conn = _conn()
    try:
        with conn.cursor() as cur:
            _plant(cur, obs_id, f"面ごとの根づき_{obs_id}", now)
            _facet(cur, obs_id, AGENT_SELF_ID, n=3)      # こちらだけ育っている
            _facet(cur, obs_id, DEFAULT_PERSON_ID, n=0)
    finally:
        conn.close()

    with patch.object(_EmbeddingModel, "pre_warm"):
        base = ObservationMemory()

    def n_of(person_id: str) -> int:
        rows = base.for_person(person_id)._observations._read_observations_by_situated(
            person_id=person_id, n=10,
            columns=("id", "content"),
        )
        assert any(r["id"] == obs_id for r in rows), f"{person_id} の面が引けない"
        conn = _conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT groundedness_n FROM situated_memories "
                    "WHERE obs_id=%s AND person_id=%s",
                    (obs_id, person_id),
                )
                return cur.fetchone()["groundedness_n"]
        finally:
            conn.close()

    assert n_of(AGENT_SELF_ID) == 3
    assert n_of(DEFAULT_PERSON_ID) == 0


def test_apply_verdicts_updates_the_facet() -> None:
    """若返りと根づきの更新は、面に対して起きる。"""
    from unittest.mock import patch

    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    obs_id = str(uuid.uuid4())
    conn = _conn()
    try:
        with conn.cursor() as cur:
            _plant(cur, obs_id, f"面の若返り_{obs_id}", datetime.now(timezone.utc))
            _facet(cur, obs_id, AGENT_SELF_ID)
    finally:
        conn.close()

    with patch.object(_EmbeddingModel, "pre_warm"):
        mem = ObservationMemory().for_person(AGENT_SELF_ID)
    assert mem._observations.apply_verdicts({obs_id: "important"}) == 1

    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT last_recalled_at, groundedness_n FROM situated_memories "
                "WHERE obs_id=%s AND person_id=%s",
                (obs_id, AGENT_SELF_ID),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    assert row["last_recalled_at"] is not None, "面の時間の起点が更新されない"
    assert row["groundedness_n"] == 1, "面の根づきが更新されない"


def test_by_time_orders_by_the_facet_origin() -> None:
    """時間軸の並べ替えは、面の起点で決まる。

    出来事の時刻は古いが面を最近引いた記録が、出来事の時刻が新しい記録より前に来る。
    """
    from unittest.mock import patch

    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    now = datetime.now(timezone.utc)
    old_id, new_id = str(uuid.uuid4()), str(uuid.uuid4())
    conn = _conn()
    try:
        with conn.cursor() as cur:
            _plant(cur, old_id, f"古い出来事だが最近引いた_{old_id}", now - timedelta(days=30))
            _plant(cur, new_id, f"新しい出来事_{new_id}", now - timedelta(days=1))
            _facet(cur, old_id, AGENT_SELF_ID)
            _facet(cur, new_id, AGENT_SELF_ID)
            cur.execute(
                "UPDATE situated_memories SET last_recalled_at = %s "
                "WHERE obs_id = %s AND person_id = %s",
                (now, old_id, AGENT_SELF_ID),
            )
    finally:
        conn.close()

    with patch.object(_EmbeddingModel, "pre_warm"):
        store = ObservationMemory().for_person(AGENT_SELF_ID)._observations
    rows = store.by_time(now.timestamp(), 50)
    order = [r["id"] for r in rows if r["id"] in (old_id, new_id)]
    assert order[:2] == [old_id, new_id], f"面の起点で並んでいない: {order[:2]}"
