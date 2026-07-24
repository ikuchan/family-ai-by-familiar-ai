"""拡散想起スライス3 (A) 共起辺：WRDB から共起≥2 の WR の要素 MI を候補で返す。

現 W 自身と自己認識 MI（self_model）は除く。seed 最遠の選別は後続。
"""

from __future__ import annotations

import os
import uuid

import psycopg2

from familiar_agent.diffuse_store import cooccurring_mi_ids
from familiar_agent.wr_store import save_wr


def _conn():
    c = psycopg2.connect(os.environ["DATABASE_URL"])
    c.autocommit = True
    return c


def _obs(cur, oid, kind="conversation"):
    cur.execute(
        "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, "
        "participants_json, scope) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
        (oid, "c", "2020-01-01", "会話", kind, "neutral", "[]", "speaker"),
    )


def test_cooccurring_returns_new_mi_excluding_w_and_self_model():
    conn = _conn()
    t = uuid.uuid4().hex[:8]
    a, b, c = f"{t}-a", f"{t}-b", f"{t}-c"       # 現 W の要素
    d = str(uuid.uuid4())                          # 共起で連想したい新 MI（会話）
    s = str(uuid.uuid4())                          # self_model（除外されるべき）
    with conn.cursor() as cur:
        _obs(cur, d, kind="conversation")
        _obs(cur, s, kind="self_model")
    save_wr(conn, [a, b, c, d])   # W と a,b,c を共有（共起3）→ d を寄与
    save_wr(conn, [b, c, s])      # W と b,c を共有（共起2）→ s を寄与（self_model で除外）
    save_wr(conn, [b, "z1", "z2"])  # 共有 b のみ（共起1）→ 対象外

    got = cooccurring_mi_ids(conn, [a, b, c], min_shared=2, limit=20)
    conn.close()
    assert d in got            # 共起≥2 の WR の新要素を拾う
    assert s not in got        # self_model は除外
    assert a not in got and b not in got and c not in got  # 現 W 自身は除外


def test_cooccurring_empty_when_w_too_small():
    conn = _conn()
    assert cooccurring_mi_ids(conn, ["only-one"], min_shared=2) == []
    conn.close()


def test_diffuse_extend_adds_cooccurring_with_a0_zero():
    """recall 結線グルー：(A)共起の候補を a0=0 で W へ足す（埋め込み不要・実DB）。"""
    from familiar_agent.config import MemoryConfig
    from familiar_agent.db import get_db
    from familiar_agent.tools.memory import ObservationMemory

    conn = _conn()
    t = uuid.uuid4().hex[:8]
    a, b = f"{t}-a", f"{t}-b"       # 現 W
    d = str(uuid.uuid4())            # 共起で連想したい新 MI
    with conn.cursor() as cur:
        _obs(cur, d, kind="conversation")
    save_wr(conn, [a, b, d])         # W={a,b} と a,b を共有（共起2）→ d を寄与
    conn.close()

    mem = ObservationMemory.__new__(ObservationMemory)
    db = get_db()
    mem._db = db
    mem._db_lock = db.lock
    mem._person_id = "SPEAKER"
    cfg = MemoryConfig()
    cfg.diffuse_max_add = 4
    cfg.diffuse_max_depth = 2

    extra = mem._diffuse_extend([{"memory_id": a}, {"memory_id": b}], cfg)
    ids = [e["memory_id"] for e in extra]
    assert d in ids                                  # 共起候補を W へ足す
    assert all(e["score"] == 0.0 for e in extra)     # a0=0（重み0）
    assert all(e["retrieval_method"] == "diffuse" for e in extra)
