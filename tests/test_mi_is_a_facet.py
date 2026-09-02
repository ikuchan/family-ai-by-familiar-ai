"""MI を「記憶 × 面」にする（案3・2026-09-02）。

**044 で situated は索引から記憶になった。** 面ごとの言葉（`content`）・時間の起点
（`last_recalled_at`）・根づきの `n` が面へ移り、想起も面のベクトルで探すようになった。
それなのに MI は「観測1行」を指したままで、想起は `DISTINCT ON (o.id)` で**どの面で
当たったかを捨てていた**。面の言葉は 2149 行に入っているのに、1 箇所も読まれていなかった。

**表は分けたままにする。** `superseded_by` が `observations` の列にしかないことが、
「**畳んでも面は残る**」を構造として保証している（版チェーンの前提・`求めの版チェーン`）。
1表へ統合すると、その保証が運用の約束へ落ちる。だから物理設計は変えず、**MI の同定だけを
面へ移す**。

**出来事ごとの量と面ごとの量を混ぜない。**

| 量 | どちらのもの |
|---|---|
| `content` | **面**（`actor` は面が持たないので出来事の本文） |
| 時間の起点 `last_recalled_at`・根づきの `n` | **面** |
| 取込の驚き `a0`・`timestamp`・`superseded_by`・`parent_id` | **出来事** |

**`memory_id` は観測 id のままにする。** 拡散想起の種・除外・supersede・WR の記録が
すべて観測 id で動いているためで、面の同定は `facet_id` として別に載せる。
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


def _vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=_DIM).astype(np.float32)
    return v / (float(np.linalg.norm(v)) or 1.0)


def _vec_sql(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def _store():
    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory().for_person(AGENT_SELF_ID)._observations


def _plant_two_facets(body: str, vec: np.ndarray) -> str:
    """1つの観測に、同じ視点（`__self__`）から2つの面を立てる。"""
    obs_id = str(uuid.uuid4())
    conn = _conn()
    try:
        with conn.cursor() as cur:
            # `emotion_vec` も入れる。`by_emotion` は感情軸を持たない観測を見ないので、
            # 入れないと「面ごとに返るか」を確かめる前に母集合が空になる。
            cur.execute(
                "INSERT INTO observations "
                "(id, content, timestamp, direction, kind, emotion, emotion_vec) "
                "VALUES (%s,%s,now(),%s,%s,%s,%s::vector)",
                (obs_id, body, "会話", "conversation", "neutral", "[0,0,0,0]"),
            )
            for key, content in (("actor", None), ("about", f"[自分のこと] {body}")):
                cur.execute(
                    "INSERT INTO situated_memories "
                    "(id, obs_id, person_id, vector, relation_key, content) "
                    "VALUES (%s,%s,%s,%s::vector,%s,%s)",
                    (str(uuid.uuid4()), obs_id, AGENT_SELF_ID, _vec_sql(vec), key, content),
                )
    finally:
        conn.close()
    return obs_id


# ── ① 面ごとに返る ──────────────────────────────────────────────────────────

def test_by_vector_returns_one_row_per_facet() -> None:
    """同じ出来事の2つの面が、それぞれ独立して返る。

    `DISTINCT ON (o.id)` が捨てていたものである。同じ出来事に2つの関わり方で触れたことは、
    畳んで消してよい情報ではない。
    """
    v = _vec(11)
    obs_id = _plant_two_facets(f"面ごとに返る {uuid.uuid4()}", v)

    rows = _store().by_vector(_vec_sql(v), 50)
    mine = [r for r in rows if r["id"] == obs_id]

    assert len(mine) == 2, f"面ごとに返っていない（{len(mine)} 行）"
    assert {r["relation_key"] for r in mine} == {"actor", "about"}
    assert len({r["facet_id"] for r in mine}) == 2, "面の id が別々でない"


def test_by_emotion_returns_one_row_per_facet() -> None:
    v = _vec(12)
    obs_id = _plant_two_facets(f"感情軸でも面ごと {uuid.uuid4()}", v)

    # 感情軸は PAD の4次元（`pad_to_search_vector`）で、記憶のベクトルとは別の空間。
    rows = _store().by_emotion("[0,0,0,0]", 50)
    mine = [r for r in rows if r["id"] == obs_id]
    assert len(mine) == 2, f"面ごとに返っていない（{len(mine)} 行）"


def test_by_time_returns_one_row_per_facet() -> None:
    import time

    v = _vec(13)
    obs_id = _plant_two_facets(f"時間軸でも面ごと {uuid.uuid4()}", v)

    rows = _store().by_time(time.time(), 50)
    mine = [r for r in rows if r["id"] == obs_id]
    assert len(mine) == 2, f"面ごとに返っていない（{len(mine)} 行）"


# ── ② 面の言葉が出る ────────────────────────────────────────────────────────

def test_the_facet_speaks_with_its_own_words() -> None:
    """`about` の面は `[自分のこと] …`、`actor` の面は素の本文。"""
    v = _vec(14)
    body = f"面の言葉 {uuid.uuid4()}"
    obs_id = _plant_two_facets(body, v)

    rows = {r["relation_key"]: r for r in _store().by_vector(_vec_sql(v), 50)
            if r["id"] == obs_id}

    assert rows["about"]["content"] == f"[自分のこと] {body}"
    assert rows["actor"]["content"] == body, "面が言葉を持たないときは出来事の本文"


# ── ③④ 出来事の量と面の量を混ぜない ────────────────────────────────────────

def test_the_event_level_quantities_are_shared_by_every_facet() -> None:
    """`groundedness_g0`（取込の驚き）は出来事のもの。どの面でも同じ値になる。"""
    v = _vec(15)
    obs_id = _plant_two_facets(f"出来事の量 {uuid.uuid4()}", v)

    mine = [r for r in _store().by_vector(_vec_sql(v), 50) if r["id"] == obs_id]
    assert len({r["groundedness_g0"] for r in mine}) == 1


def test_the_facet_level_quantities_differ_per_facet() -> None:
    """根づきの `n` は面のもの。同じ出来事でも面ごとに違う値を持てる。"""
    v = _vec(16)
    obs_id = _plant_two_facets(f"面の量 {uuid.uuid4()}", v)
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE situated_memories SET groundedness_n = 3 "
                "WHERE obs_id = %s AND relation_key = 'about'",
                (obs_id,),
            )
    finally:
        conn.close()

    mine = {r["relation_key"]: r for r in _store().by_vector(_vec_sql(v), 50)
            if r["id"] == obs_id}
    assert mine["about"]["groundedness_n"] == 3
    assert mine["actor"]["groundedness_n"] == 0


# ── ⑤ MI が面の同定を持ち、視点3属性を持たない ──────────────────────────────

def test_mi_identifies_a_facet() -> None:
    from familiar_agent.io.oif import _to_recalled

    row = {
        "memory_id": "obs-1", "facet_id": "facet-1",
        "person_id": AGENT_SELF_ID, "relation_key": "about",
        "summary": "[自分のこと] ねこの話", "timestamp": None, "direction": "会話",
        "emotion": "neutral", "groundedness_g0": 0.8, "groundedness_n": 2,
        "fit": 0.5, "groundedness": 0.3,
    }
    mi = _to_recalled(row).mi

    assert mi.id == "facet-1", "MI の id は面の id"
    assert mi.obs_id == "obs-1", "どの出来事の面かも持つ"
    assert mi.person_id == AGENT_SELF_ID
    assert mi.relation_key == "about"
    assert mi.groundedness_n == 2, "面ごとの量が載っている"
    assert mi.groundedness_g0 == 0.8, "出来事ごとの量も載っている"


def test_mi_has_no_perspective_columns() -> None:
    """視点3属性は面が引き取った（段5 の前提）。"""
    import dataclasses

    from familiar_agent.io.oif import MI

    names = {f.name for f in dataclasses.fields(MI)}
    assert not (names & {"writer_id", "subject_id", "participants"}), sorted(names)
