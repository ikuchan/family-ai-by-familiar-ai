"""拡張想起の (B) エンティティ辺を situated の面へ付け替える（段4）。

(B) 辺だけが、まだ `observations` の視点列3つ（`subject_id`／`participants_json`／
`writer_id`）を読んでいた。人と記憶の結びつきは situated が担う（[D-在席相関/V2]）ので、
種の抽出も母集合も面から引く。これが段5（視点列3つの撤去）の前提になる。

**役割の対応**は 047 が定めたとおり。`about`（話題の主体）／`present`（そばに居た）／
`actor`（誰がやったか）。種の優先順もこの順である。**047 が機械で立てるのは `actor` と
`present` だけ**なので、`about` は REST 内省（記-a-ほ）が足すまで 0 行のまま動く。

**共通の記憶**を (B) 辺へ足す。在席者が2人以上いるとき、その**全員**と関係を持つ観測は、
その場に居合わせた人たちで共有している出来事である。片方としか関係の無い観測は入れない。
2026-08-21 のダンプでは、実在の人2人以上と関係行を持つ観測が 190 件あった。
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import psycopg2

from familiar_agent.core.diffuse import select_entity_seeds
from familiar_agent.diffuse_store import (
    fetch_relation_persons,
    recall_by_person,
    shared_memory_ids,
)

_DB_URL = os.environ["DATABASE_URL"]
_DIM = 1024


def _conn():
    # 本番の `Database.conn()` と同じ素の接続（タプル行）。`diffuse_store` はこれを受け取る。
    c = psycopg2.connect(_DB_URL)
    c.autocommit = True
    return c


def _vec_sql(seed: int) -> str:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=_DIM).astype(np.float32)
    v /= float(np.linalg.norm(v)) or 1.0
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def _obs(cur, ts: str = "2020-01-01", superseded: bool = False) -> str:
    oid = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, superseded_by) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s)",
        (oid, f"段4 {oid}", ts, "会話", "conversation", "neutral",
         (str(uuid.uuid4()) if superseded else None)),
    )
    return oid


def _facet(cur, obs_id: str, person_id: str, relation_key: str) -> None:
    cur.execute(
        "INSERT INTO situated_memories (id, obs_id, person_id, vector, relation_key) "
        "VALUES (%s,%s,%s,%s::vector,%s)",
        (str(uuid.uuid4()), obs_id, person_id, _vec_sql(hash(obs_id) % 9973), relation_key),
    )


def _person(cur, pid: str) -> None:
    cur.execute(
        "INSERT INTO persons (id, name, created_at, updated_at) "
        "VALUES (%s,%s,now(),now()) ON CONFLICT (id) DO NOTHING",
        (pid, f"n-{pid[:8]}"),
    )


# ── ① 種は面から取り、`about`→`present`→`actor` の順に並ぶ ──────────────────

def test_seeds_come_from_facets_in_role_order() -> None:
    """種の優先順は「話題の主体 → そばに居た → やった人」。"""
    tag = uuid.uuid4().hex[:8]
    A, P, W = f"about-{tag}", f"present-{tag}", f"actor-{tag}"
    conn = _conn()
    try:
        with conn.cursor() as cur:
            for pid in (A, P, W):
                _person(cur, pid)
            oid = _obs(cur)
            _facet(cur, oid, W, "actor")
            _facet(cur, oid, P, "present")
            _facet(cur, oid, A, "about")
        rows = fetch_relation_persons(conn, [oid])
    finally:
        conn.close()

    assert select_entity_seeds(rows, exclude=set()) == [A, P, W]


def test_the_diffuse_edge_no_longer_reads_the_perspective_columns() -> None:
    """視点列3つを **SQL で** 読まなくなった（段5 の前提）。

    説明文での言及は数えない。数えるのは実行される問い合わせだけである。
    """
    import ast
    import inspect
    import re

    from familiar_agent import diffuse_store
    from familiar_agent.core import diffuse

    def sql_literals(mod) -> list[str]:
        """SQL らしき文字列リテラルを、連結を畳んで集める。"""
        out = []

        def flat(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                left, right = flat(node.left), flat(node.right)
                return (left + right) if (left is not None and right is not None) else None
            return None

        for node in ast.walk(ast.parse(inspect.getsource(mod))):
            text = flat(node)
            if text and re.search(r"\b(SELECT|INSERT|UPDATE|DELETE)\b", text, re.I):
                out.append(" ".join(text.split()))
        return out

    for mod in (diffuse_store, diffuse):
        for sql in sql_literals(mod):
            for col in ("subject_id", "participants_json", "writer_id"):
                assert col not in sql, f"{mod.__name__} の SQL がまだ {col} を読んでいる：{sql[:90]}"


# ── ② 母集合は `about` と `present` の面。`actor` だけの観測は入らない ───────

def test_the_pool_is_about_and_present_not_actor() -> None:
    """その人が「やった」だけの記録は、その人を種にした再想起の母集合に入れない。

    `actor` を入れると、パジュ自身が書いた記録（`actor` が `__self__` の 6433 行）が
    どの種からも湧いてしまう。種として使うのと、母集合にするのは別の問いである。
    """
    tag = uuid.uuid4().hex[:8]
    X = f"pool-{tag}"
    conn = _conn()
    try:
        with conn.cursor() as cur:
            _person(cur, X)
            about = _obs(cur, ts="2020-01-01")
            present = _obs(cur, ts="2025-01-01")
            actor_only = _obs(cur, ts="2026-01-01")
            gone = _obs(cur, ts="2026-02-01", superseded=True)
            _facet(cur, about, X, "about")
            _facet(cur, present, X, "present")
            _facet(cur, actor_only, X, "actor")
            _facet(cur, gone, X, "present")
        got = recall_by_person(conn, X, limit=10)
    finally:
        conn.close()

    assert set(got) == {about, present}, "actor だけの観測か、畳まれた版が混じっている"
    assert got[0] == present, "新しい順でない"


# ── ③④ 共通の記憶：在席者2人**以上**が、ともに関係を持つ観測 ────────────────

def test_shared_memory_needs_every_present_person() -> None:
    """二人がともに関係を持つ観測だけを返す。片方だけの観測は返さない。"""
    tag = uuid.uuid4().hex[:8]
    X, Y = f"sx-{tag}", f"sy-{tag}"
    conn = _conn()
    try:
        with conn.cursor() as cur:
            _person(cur, X); _person(cur, Y)
            both = _obs(cur, ts="2025-01-01")
            only_x = _obs(cur, ts="2025-02-01")
            _facet(cur, both, X, "present")
            _facet(cur, both, Y, "about")
            _facet(cur, only_x, X, "present")
        got = shared_memory_ids(conn, [X, Y], limit=10)
    finally:
        conn.close()

    assert got == [both], "片方としか関係の無い観測が混じっている"


def test_shared_memory_is_empty_for_a_single_person() -> None:
    """在席者が一人なら「共通の記憶」は無い（反証側）。"""
    tag = uuid.uuid4().hex[:8]
    X = f"solo-{tag}"
    conn = _conn()
    try:
        with conn.cursor() as cur:
            _person(cur, X)
            oid = _obs(cur)
            _facet(cur, oid, X, "present")
        got = shared_memory_ids(conn, [X], limit=10)
    finally:
        conn.close()

    assert got == []
