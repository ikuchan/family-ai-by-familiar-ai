"""047（関係エッジの機械的な生成）と 048（内なる記録はエージェントのもの）。

**面の生成は二段である。**

  段① 機械（047・048）  `actor`（誰がやったか）・`present`（誰が居たか）
                         視点列から確実に出る。全観測に土台として立つ。
  段② REST 内省（記-a）  `addressee`／`about`／`experiencer`／`beneficiary`／
                         `companion`／`source`／`owner` …
                         本文を読んで意味役割を抽出し、**既存の観測にもさかのぼって**
                         面を足していく。

047 は段①だけを実装する。段②は記-a の仕事で、ここでは立てない。

**生成が「全登録人物」から「関係のある人だけ」へ変わる。** これまでは観測1件につき
登録人物全員＋AGENT_SELF の行を `presence` 固定で作っていた（6433×6≈38,600 行）。
これからは `actor` が観測1件につき1行、`present` が在席者ぶんだけ立つ（≈6,806 行）。

**面の `content` は `[役割の札] ` ＋ 出来事の本文である**（実物で全役割 厳密一致）。
`actor` だけ content を持たない（全観測に立つので書き直す意味がない）。
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import patch

import numpy as np
import psycopg2
import psycopg2.extras

from familiar_agent.person_memory_manager import AGENT_SELF_ID, DEFAULT_PERSON_ID

_DB_URL = os.environ["DATABASE_URL"]


def _conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _mem():
    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory()


def _person(cur, name: str) -> str:
    pid = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO persons (id, name, display_name, created_at, updated_at) "
        "VALUES (%s, %s, %s, now(), now())",
        (pid, name, name),
    )
    return pid


def _plant(cur, obs_id: str, content: str, *, writer: str, participants: list[str]):
    import json

    cur.execute(
        "INSERT INTO observations "
        "(id, content, timestamp, direction, kind, emotion, person_id, writer_id, "
        " subject_id, participants_json) "
        "VALUES (%s, %s, now(), %s, %s, %s, %s, %s, %s, %s)",
        (obs_id, content, "unknown", "observation", "neutral",
         DEFAULT_PERSON_ID, writer, writer, json.dumps(participants)),
    )


def _facets(obs_id: str) -> list[dict]:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT person_id, relation_key, content FROM situated_memories "
                "WHERE obs_id = %s ORDER BY relation_key, person_id",
                (obs_id,),
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


# ── 047：機械的な2役割 ─────────────────────────────────────────────

def test_actor_is_one_row_per_observation() -> None:
    """`actor` は観測1件につき1行、`writer_id` の人に立つ。"""
    obs_id = str(uuid.uuid4())
    conn = _conn()
    try:
        with conn.cursor() as cur:
            who = _person(cur, f"actor試験_{obs_id[:8]}")
            _plant(cur, obs_id, f"actor テスト_{obs_id}", writer=who, participants=[])
    finally:
        conn.close()

    conn = _conn()
    try:
        _mem()._situated.refresh_situated_memories(
            conn, obs_id, np.arange(1024, dtype=np.float32))
    finally:
        conn.close()

    actor = [f for f in _facets(obs_id) if f["relation_key"] == "actor"]
    assert len(actor) == 1, f"actor が {len(actor)} 行"
    assert actor[0]["person_id"] == who
    assert actor[0]["content"] is None, "actor は content を持たない"


def test_present_is_one_row_per_participant_with_a_label() -> None:
    """`present` は在席者ぶん立ち、content は `[そばに居た] ` ＋ 本文。"""
    obs_id = str(uuid.uuid4())
    body = f"present テスト_{obs_id}"
    conn = _conn()
    try:
        with conn.cursor() as cur:
            who = _person(cur, f"actor_{obs_id[:8]}")
            a = _person(cur, f"在席A_{obs_id[:8]}")
            b = _person(cur, f"在席B_{obs_id[:8]}")
            _plant(cur, obs_id, body, writer=who, participants=[a, b])
    finally:
        conn.close()

    conn = _conn()
    try:
        _mem()._situated.refresh_situated_memories(
            conn, obs_id, np.arange(1024, dtype=np.float32))
    finally:
        conn.close()

    present = [f for f in _facets(obs_id) if f["relation_key"] == "present"]
    assert {f["person_id"] for f in present} == {a, b}
    for f in present:
        assert f["content"] == f"[そばに居た] {body}"


def test_unrelated_people_get_no_facet() -> None:
    """関係の無い人には立たない（**本体**）。

    047 の前は登録人物全員に `presence` の行を作っていたので、ここが落ちる。
    """
    obs_id = str(uuid.uuid4())
    conn = _conn()
    try:
        with conn.cursor() as cur:
            who = _person(cur, f"actor_{obs_id[:8]}")
            bystander = _person(cur, f"無関係_{obs_id[:8]}")
            _plant(cur, obs_id, f"無関係テスト_{obs_id}", writer=who, participants=[])
    finally:
        conn.close()

    conn = _conn()
    try:
        _mem()._situated.refresh_situated_memories(
            conn, obs_id, np.arange(1024, dtype=np.float32))
    finally:
        conn.close()

    people = {f["person_id"] for f in _facets(obs_id)}
    assert bystander not in people, "関係の無い人に面が立っている"
    assert people == {who}, f"立った面: {people}"


def test_the_relation_key_default_is_present() -> None:
    """既定の関係名は `present`（022 の `presence` から改めた）。"""
    import inspect

    from familiar_agent.store.situated import SituatedVectors

    sig = inspect.signature(SituatedVectors._upsert_situated_embedding)
    assert sig.parameters["relation_key"].default == "present"


# ── 048：内なる記録はエージェントのもの ─────────────────────────────

def test_inner_records_belong_to_the_agent() -> None:
    """`writer_id` が `default`（話者未解決）なら `actor` は `__self__`。

    047 が `writer_id` から素直に作ると `default` の actor が立つ。実物では
    `default` の面は1件も無く、2544 件がすべて `__self__` へ寄せられていた。
    """
    obs_id = str(uuid.uuid4())
    conn = _conn()
    try:
        with conn.cursor() as cur:
            _plant(cur, obs_id, f"内なる記録_{obs_id}",
                   writer=DEFAULT_PERSON_ID, participants=[])
    finally:
        conn.close()

    conn = _conn()
    try:
        _mem()._situated.refresh_situated_memories(
            conn, obs_id, np.arange(1024, dtype=np.float32))
    finally:
        conn.close()

    actor = [f for f in _facets(obs_id) if f["relation_key"] == "actor"]
    assert len(actor) == 1
    assert actor[0]["person_id"] == AGENT_SELF_ID, (
        f"内なる記録の actor が {actor[0]['person_id']}（`default` のままになっている）"
    )


# ── 想起：1観測1候補へ畳む ─────────────────────────────────────────

def test_recall_returns_one_candidate_per_observation() -> None:
    """同じ観測に複数の面が立っても、候補には1回しか出ない。

    047 で `actor` と `present` が同じ人に立ちうるようになる（自分が在席者に
    含まれるとき）。畳まないと K=7 の枠を1つの記憶が複数食う。
    """
    obs_id = str(uuid.uuid4())
    vec = "[" + ",".join(["1"] + ["0"] * 1023) + "]"
    conn = _conn()
    try:
        with conn.cursor() as cur:
            _plant(cur, obs_id, f"畳み込みテスト_{obs_id}",
                   writer=AGENT_SELF_ID, participants=[AGENT_SELF_ID])
            for key in ("actor", "present"):
                cur.execute(
                    "INSERT INTO situated_memories "
                    "(id, obs_id, person_id, vector, relation_key) "
                    "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    (str(uuid.uuid4()), obs_id, AGENT_SELF_ID, vec, key),
                )
    finally:
        conn.close()

    store = _mem().for_person(AGENT_SELF_ID)._observations
    rows = store.by_vector(vec, 50, kind=None, exclude_ids=None)
    hits = [r for r in rows if str(r["id"]) == obs_id]
    assert len(hits) == 1, f"同じ観測が候補に {len(hits)} 回出ている"


def test_situated_cosines_takes_the_strongest_facet() -> None:
    """複数の面があるとき、最も強く結びつく面で代表させる（max）。

    いまは dict の最後の行で上書きされ、どの面が採られるかが DB の返す順に依存する。
    """
    obs_id = str(uuid.uuid4())
    near = "[" + ",".join(["1"] + ["0"] * 1023) + "]"
    far = "[" + ",".join(["0", "1"] + ["0"] * 1022) + "]"
    conn = _conn()
    try:
        with conn.cursor() as cur:
            _plant(cur, obs_id, f"max テスト_{obs_id}",
                   writer=AGENT_SELF_ID, participants=[])
            for key, v in (("actor", far), ("present", near)):
                cur.execute(
                    "INSERT INTO situated_memories "
                    "(id, obs_id, person_id, vector, relation_key) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (str(uuid.uuid4()), obs_id, AGENT_SELF_ID, v, key),
                )
    finally:
        conn.close()

    got = _mem()._observations.situated_cosines(near, [obs_id], AGENT_SELF_ID)
    assert obs_id in got
    assert got[obs_id] > 0.99, f"最も近い面が採られていない: {got[obs_id]}"
