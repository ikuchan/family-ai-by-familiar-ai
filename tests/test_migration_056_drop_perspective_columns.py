"""056：`observations` から視点列3つを落とす（段5）。

`writer_id`（誰が書いたか）・`subject_id`（誰についてか）・`participants_json`（誰が居たか）
を撤去する。人と記憶の結びつきは situated だけが担う（[D-在席相関/V2]）。

**面を立てる材料そのものは消さない。** 誰がしたこと・誰が居たかは書き込みの瞬間には要る
情報で、落とすのは「観測の行に残しておくこと」だけである。材料は引数で渡し、立った面が
以後の正になる。

**`subject_id` は写さずに落とせる。** 2026-08-21 のダンプで実在の人を指すのは 397 件だが、
**その全件がその人の面を既に持っている**（`present` 337／`about` 79／`addressee` 35／
`actor` 26／`source` 9／`beneficiary` 2）。写す先が無い。

**重複判定は `actor` の面へ移す。** 重複とは「同じ書き手が同じ内容を同じ kind で窓の内に」
であって、家族の二人が同じ挨拶をしたものは重複ではない（042 で `writer_id` へ移したものを、
列の撤去に合わせて面へ移す）。
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import psycopg2
import psycopg2.extras
from unittest.mock import patch

from familiar_agent.person_memory_manager import AGENT_SELF_ID

_DB_URL = os.environ["DATABASE_URL"]
_DIM = 1024


def _conn():
    c = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    c.autocommit = True
    return c


def _mem():
    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory()


def _facets(obs_id: str) -> dict[tuple[str, str], dict]:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT person_id, relation_key, content, vector::text AS v "
                "FROM situated_memories WHERE obs_id = %s",
                (obs_id,),
            )
            return {(r["person_id"], r["relation_key"]): dict(r) for r in cur.fetchall()}
    finally:
        conn.close()


# ── ① 列が消えた ────────────────────────────────────────────────────────────

def test_the_three_perspective_columns_are_gone() -> None:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'observations'"
            )
            columns = {r["column_name"] for r in cur.fetchall()}
    finally:
        conn.close()
    for col in ("writer_id", "subject_id", "participants_json"):
        assert col not in columns, f"{col} が残っている"


# ── ② 重複判定は書き手の面で効く ────────────────────────────────────────────

def test_dedup_folds_the_same_writer_and_keeps_different_writers() -> None:
    """同じ人が窓の内に同じことを書けば畳み、別の人なら畳まない。

    **別の人は実在の人で試す。** `default` は「話者がまだ分からない」の置き場であって
    人ではなく、048 の規則でその `actor` は `__self__` になる。`default` と `__self__` を
    別人として使うと、畳まれて当然のものを「畳まれた」と読み違える。
    """
    who = str(uuid.uuid4())
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO persons (id, name, created_at, updated_at) "
                "VALUES (%s,%s,now(),now()) ON CONFLICT (id) DO NOTHING",
                (who, f"n-{who[:8]}"),
            )
    finally:
        conn.close()

    mem = _mem()
    original = os.environ.get("MEMORY_DEDUP_WINDOW_SECS")
    os.environ["MEMORY_DEDUP_WINDOW_SECS"] = "30"
    same = f"同じ書き手_{uuid.uuid4()}"
    other = f"別の書き手_{uuid.uuid4()}"
    try:
        mem.save_with_id(same, kind="utterance", writer_id=AGENT_SELF_ID)
        mem.save_with_id(same, kind="utterance", writer_id=AGENT_SELF_ID)
        mem.save_with_id(other, kind="utterance", writer_id=AGENT_SELF_ID)
        mem.save_with_id(other, kind="utterance", writer_id=who)
    finally:
        if original is None:
            os.environ.pop("MEMORY_DEDUP_WINDOW_SECS", None)
        else:
            os.environ["MEMORY_DEDUP_WINDOW_SECS"] = original

    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT content, count(*) AS n FROM observations "
                "WHERE content IN (%s, %s) AND superseded_by IS NULL GROUP BY 1",
                (same, other),
            )
            got = {r["content"]: r["n"] for r in cur.fetchall()}
    finally:
        conn.close()
    assert got[same] == 1, f"同じ書き手の重複が畳まれていない（{got[same]} 行）"
    assert got[other] == 2, f"別の書き手まで畳まれた（{got[other]} 行）"


# ── ③④ 面の材料は引数で渡る ────────────────────────────────────────────────

def test_the_facet_builder_does_not_read_the_observation_row() -> None:
    """面を立てるのに `observations` を読み直さない（材料は引数で来る）。"""
    import ast
    import inspect
    import re
    import textwrap

    from familiar_agent.store.situated import SituatedVectors

    src = textwrap.dedent(inspect.getsource(SituatedVectors.refresh_situated_memories))

    def flat(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = flat(node.left), flat(node.right)
            return (left + right) if (left is not None and right is not None) else None
        return None

    for node in ast.walk(ast.parse(src)):
        text = flat(node)
        if text and re.search(r"SELECT\b.*\bFROM\s+observations\b", " ".join(text.split()), re.I):
            raise AssertionError(f"観測を読み直している：{' '.join(text.split())[:90]}")


def test_the_facets_stand_from_the_arguments_alone() -> None:
    """引数だけで `actor` と `present` の面が立つ。"""
    obs_id = str(uuid.uuid4())
    body = f"引数から面を立てる {obs_id}"
    other = str(uuid.uuid4())
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO persons (id, name, created_at, updated_at) "
                "VALUES (%s,%s,now(),now()) ON CONFLICT (id) DO NOTHING",
                (other, f"n-{other[:8]}"),
            )
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction, kind, emotion) "
                "VALUES (%s,%s,now(),%s,%s,%s)",
                (obs_id, body, "会話", "conversation", "neutral"),
            )
    finally:
        conn.close()

    mem = _mem()
    with mem._db.lock:
        conn2 = mem._db.conn()
        mem._situated.refresh_situated_memories(
            conn2, obs_id, np.ones(_DIM, dtype=np.float32),
            body=body, writer_id=AGENT_SELF_ID, participants=[other],
        )
        conn2.commit()

    got = _facets(obs_id)
    assert (AGENT_SELF_ID, "actor") in got
    assert (other, "present") in got
    assert got[(other, "present")]["content"] == f"[そばに居た] {body}"


# ── ⑤ 本文の更新は面をなぞる ────────────────────────────────────────────────

def test_appending_refreshes_every_facet_without_dropping_the_semantic_ones() -> None:
    """本文を足すと、REST が足した意味役割の面もベクトルが新しくなり、面は消えない。

    いままでは `actor` と `present` を作り直すだけだったので、**REST が足した面の
    ベクトルは古い本文のまま取り残されていた**。面が正なので、なぞって更新する。
    """
    obs_id = str(uuid.uuid4())
    body = f"面をなぞる {obs_id}"
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction, kind, emotion) "
                "VALUES (%s,%s,now(),%s,%s,%s)",
                (obs_id, body, "会話", "conversation", "neutral"),
            )
            for key, content in (("actor", None),
                                 ("about", f"[自分のこと] {body}")):
                cur.execute(
                    "INSERT INTO situated_memories "
                    "(id, obs_id, person_id, vector, relation_key, content) "
                    "VALUES (%s,%s,%s,%s::vector,%s,%s)",
                    (str(uuid.uuid4()), obs_id, AGENT_SELF_ID,
                     "[" + ",".join(["0.01"] * _DIM) + "]", key, content),
                )
    finally:
        conn.close()

    before = _facets(obs_id)
    assert _mem()._observations.append_and_reembed(obs_id, "あとから足した一行") is True
    after = _facets(obs_id)

    assert set(before) == set(after), "面が消えた、または増えた"
    assert after[(AGENT_SELF_ID, "about")]["v"] != before[(AGENT_SELF_ID, "about")]["v"], \
        "REST が足した面のベクトルが古いまま"
    assert after[(AGENT_SELF_ID, "actor")]["v"] != before[(AGENT_SELF_ID, "actor")]["v"]
    assert after[(AGENT_SELF_ID, "about")]["content"] == f"[自分のこと] {body}", \
        "REST が書いた言葉を機械が書き換えている"
