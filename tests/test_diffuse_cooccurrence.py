"""拡散想起スライス3 (A) 共起辺：共起≥2 の集まりの要素を候補で返す。

現 W 自身と自己認識 MI（self_model）は除く。seed 最遠の選別は後続。

器は関係へ移した（段 5・`設計方針_MI間の関係`）。`wr_records` は
`relations(kind='共起')` になり、読み書きは `RelationStore` が持つ。
"""

from __future__ import annotations

import os
import uuid

import psycopg2

from familiar_agent.db import get_db
from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID
from familiar_agent.store.context import StoreContext
from familiar_agent.store.relations import RelationStore


def _store() -> RelationStore:
    db = get_db()
    return RelationStore(
        StoreContext(db=db, lock=db.lock, person_id=DEFAULT_PERSON_ID, embedder=None)
    )


def _conn():
    c = psycopg2.connect(os.environ["DATABASE_URL"])
    c.autocommit = True
    return c


def _obs(cur, oid, kind="conversation"):
    cur.execute(
        "INSERT INTO observations (id, content, timestamp, direction, kind, emotion) "
        "VALUES (%s,%s,%s,%s,%s,%s)",
        (oid, "c", "2020-01-01", "会話", kind, "neutral"),
    )


def test_cooccurring_returns_new_mi_excluding_w_and_self_model():
    conn = _conn()
    t = uuid.uuid4().hex[:8]
    a, b, c = f"{t}-a", f"{t}-b", f"{t}-c"  # 現 W の要素
    d = str(uuid.uuid4())  # 共起で連想したい新 MI（会話）
    s = str(uuid.uuid4())  # self_model（除外されるべき）
    with conn.cursor() as cur:
        _obs(cur, d, kind="conversation")
        _obs(cur, s, kind="self_model")
    store = _store()
    store.record_cooccurrence([a, b, c, d])  # a,b,c を共有（共起3）→ d を寄与
    store.record_cooccurrence([b, c, s])  # b,c を共有（共起2）→ s は self_model で除外
    store.record_cooccurrence([b, "z1", "z2"])  # 共有 b のみ（共起1）→ 対象外

    got = store.cooccurring([a, b, c], min_shared=2, limit=20)
    conn.close()
    assert d in got  # 共起≥2 の WR の新要素を拾う
    assert s not in got  # self_model は除外
    assert a not in got and b not in got and c not in got  # 現 W 自身は除外


def test_cooccurring_empty_when_w_too_small():
    assert _store().cooccurring(["only-one"], min_shared=2) == []


def test_diffuse_extend_adds_cooccurring_with_a0_zero():
    """recall 結線グルー：(A)共起の候補を a0=0 で W へ足す（埋め込み不要・実DB）。"""
    from familiar_agent.config import MemoryConfig
    from familiar_agent.db import get_db
    from familiar_agent.tools.memory import ObservationMemory

    conn = _conn()
    t = uuid.uuid4().hex[:8]
    a, b = f"{t}-a", f"{t}-b"  # 現 W
    d = str(uuid.uuid4())  # 共起で連想したい新 MI
    with conn.cursor() as cur:
        _obs(cur, d, kind="conversation")
    conn.close()
    _store().record_cooccurrence([a, b, d])  # a,b を共有（共起2）→ d を寄与

    mem = ObservationMemory.__new__(ObservationMemory)
    db = get_db()
    mem._db = db
    mem._db_lock = db.lock
    mem._person_id = "SPEAKER"
    # 共起は層（`RelationStore`）から引くようになった（段 5）。層は文脈から組む。
    mem._ctx = StoreContext(db=db, lock=db.lock, person_id="SPEAKER", embedder=None)
    cfg = MemoryConfig()
    cfg.diffuse_max_add = 4
    cfg.diffuse_max_depth = 2

    extra = mem._diffuse_extend([{"memory_id": a}, {"memory_id": b}], cfg)
    ids = [e["memory_id"] for e in extra]
    assert d in ids  # 共起候補を W へ足す
    assert all(e["fit"] == 0.0 for e in extra)  # a0=0（重み0）
    assert all(e["retrieval_method"] == "diffuse" for e in extra)


def test_order_ids_by_farthest_puts_novel_first():
    """4b：seed から遠い（コサイン低い）候補を先頭へ、埋め込み無しは末尾。"""
    import numpy as np

    from familiar_agent.diffuse_store import order_ids_by_farthest

    conn = _conn()
    near = str(uuid.uuid4())  # seed とほぼ同方向（近い）
    far = str(uuid.uuid4())  # seed と直交（遠い）
    noemb = str(uuid.uuid4())  # 埋め込み無し
    with conn.cursor() as cur:
        _obs(cur, near)
        _obs(cur, far)
        _obs(cur, noemb)
        cur.execute(
            "INSERT INTO obs_embeddings (obs_id, vector) VALUES (%s,%s)",
            (near, np.array([1, 0, 0], dtype=np.float32).tobytes()),
        )
        cur.execute(
            "INSERT INTO obs_embeddings (obs_id, vector) VALUES (%s,%s)",
            (far, np.array([0, 0, 1], dtype=np.float32).tobytes()),
        )
    out = order_ids_by_farthest(conn, [near, far, noemb], np.array([1, 0, 0], dtype=np.float32))
    conn.close()
    assert out[0] == far  # 最遠（新規性高い）が先頭
    assert out[-1] == noemb  # 埋め込み無しは末尾
