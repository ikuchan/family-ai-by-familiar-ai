"""045（`persons.perspective_vec` の撤去）が効いていることを確かめる。

**ベクトルの差を「人」でなく「関係」で作る形へ移す前段である。**

044 までの situated ベクトルは `normalise(mem_vec + ALPHA·p_vec − mu)` で、`p_vec` は
人ごとに育てた視点ベクトルだった（`ALPHA=0.30`・書き込みごとに `lr=0.05` で更新）。
2026-08-03 のダンプでは 6人中3人が非NULL で、この項は生きていた。

045 で視点項を落とす。差は 047 が足す**関係項**（`relation_concept`）が担う。
実物（2026-08-21）でも、**同じ観測・同じ関係なら人が違ってもコサインは 1.000000** で
一致する（`addressee` 186 対・`about` 105 対ほか）。ベクトルは「観測 × 関係」だけで
決まり、人には依らない。

**視点シフト検索（役割1）の絞り（`s.person_id = ?`）は残る。** 変わるのは「どの行が
母集合か」ではなく「その行のベクトルが人によって違うか」である。

045 の直後・047 の前は、生成が `relation_key='presence'` 固定なので関係による差も
まだ無い。したがって同じ観測に対する全員のベクトルが同一になる。
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import psycopg2
import psycopg2.extras

from familiar_agent.person_memory_manager import AGENT_SELF_ID, DEFAULT_PERSON_ID

_DB_URL = os.environ["DATABASE_URL"]


def _conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def test_persons_has_no_perspective_vec() -> None:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name='persons'"
            )
            cols = {r["column_name"] for r in cur.fetchall()}
    finally:
        conn.close()
    assert "perspective_vec" not in cols, sorted(cols)


def test_situated_vector_no_longer_takes_a_perspective() -> None:
    """式から視点項が消える。中心化だけが残る。

    受け取るだけで使わない引数にはしない。残しておくと「効いている」と読まれる。
    """
    from familiar_agent.store.situated import _normalise, _situated_vector

    mem = np.arange(1024, dtype=np.float32)
    mu = np.ones(1024, dtype=np.float32) * 0.01
    got = _situated_vector(mem, mu)
    assert np.allclose(got, _normalise(mem - mu))
    assert np.allclose(_situated_vector(mem, None), _normalise(mem))


def test_the_write_path_no_longer_reads_who() -> None:
    """situated を作る経路が、人の視点を読まない。

    値で比べる形にすると、テスト DB では視点がどちらも NULL で差が出ず、視点項が
    生きていても通ってしまう（識別力が無い）。**読みに行かないこと**を構造で固定する。
    """
    import inspect

    from familiar_agent.store.situated import SituatedVectors, _situated_vector

    for fn in (SituatedVectors._upsert_situated_embedding, _situated_vector):
        src = inspect.getsource(fn)
        assert "perspective_vec" not in src, f"{fn.__name__} が視点を読んでいる"
        assert "ALPHA" not in src, f"{fn.__name__} に視点の係数が残っている"

    layer = inspect.getsource(SituatedVectors)
    for name in ("_get_perspective_vec", "update_perspective_vec"):
        assert f"def {name}" not in layer, f"{name} が層に残っている"


def test_the_vector_no_longer_depends_on_who() -> None:
    """同じ観測なら、人が違ってもベクトルが一致する（値での確認）。

    045 の後は「観測 × 関係」だけで決まる（045 の時点では関係も `presence` 固定なので
    観測ごとに1つになる）。実物（2026-08-21）でも同じ観測・同じ関係で人が違っても
    コサインは 1.000000 だった。
    """
    from unittest.mock import patch

    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    obs_id = str(uuid.uuid4())
    with patch.object(_EmbeddingModel, "pre_warm"):
        mem = ObservationMemory()

    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction, kind, emotion) "
                "VALUES (%s, %s, now(), %s, %s, %s)",
                (obs_id, f"視点なしテスト_{obs_id}", "unknown", "observation", "neutral"),
            )
        vec = np.arange(1024, dtype=np.float32)
        for pid in (AGENT_SELF_ID, DEFAULT_PERSON_ID):
            mem._situated._upsert_situated_embedding(conn, obs_id, pid, vec)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 - (a.vector <=> b.vector) AS cos "
                "FROM situated_memories a JOIN situated_memories b "
                "  ON a.obs_id = b.obs_id AND a.relation_key = b.relation_key "
                " AND a.person_id < b.person_id "
                "WHERE a.obs_id = %s",
                (obs_id,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    assert rows, "比べる対が作れていない"
    for r in rows:
        assert abs(float(r["cos"]) - 1.0) < 1e-6, f"人でベクトルが違う: {r['cos']}"
