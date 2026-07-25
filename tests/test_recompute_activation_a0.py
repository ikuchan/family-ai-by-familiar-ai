"""取込 a0（内容の新規性）の一括再計算。

`content_novelty` は近傍 K 件の平均コサインの裏返しで novelty を測るが、その近傍検索は
`person_id`・`superseded_by`・`kind` で絞り込んだベクトル検索であり、HNSW の絞り込み検索で
本当の近傍が取れていなかった。近傍が K 未満なら既定値、揃っても本当の近傍でなければ平均
コサインが低く出て novelty が過大になる。実機では「パパが2026年ワールドカップ決勝戦を観た」
に a0=1.000（正しくは 0.674）が付いていた。既存行を全走査の正確な近傍で計算し直す。
"""

from __future__ import annotations

import importlib.util
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]
_MIGRATION = "2026-07-25-031_recompute_activation_a0.py"
_AGENT_SELF = "00000000-0000-0000-0000-000000000000"


def _conn():
    c = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    c.autocommit = False
    return c


def _run_migration(conn) -> None:
    path = Path(__file__).parent.parent / "migration" / _MIGRATION
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.upgrade(conn)
    conn.commit()


def _plant(conn, content: str, vec: list[float], ts: datetime, a0: float) -> str:
    """観測と AGENT_SELF 視点の situated ベクトルを直に置く。"""
    oid = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO observations (id,content,timestamp,direction,kind,emotion,person_id,"
            " writer_id,subject_id,participants_json,scope,activation_a0) "
            "VALUES (%s,%s,%s,'unknown','observation','neutral',%s,%s,%s,'[]','speaker',%s)",
            (oid, content, ts, _AGENT_SELF, _AGENT_SELF, _AGENT_SELF, a0),
        )
        cur.execute(
            "INSERT INTO situated_embeddings (id, obs_id, person_id, vector) VALUES (%s,%s,%s,%s)",
            (str(uuid.uuid4()), oid, _AGENT_SELF,
             "[" + ",".join(f"{x:.6f}" for x in vec) + "]"),
        )
    conn.commit()
    return oid


def _a0(conn, oid: str) -> float:
    with conn.cursor() as cur:
        cur.execute("SELECT activation_a0 FROM observations WHERE id=%s", (oid,))
        return float(cur.fetchone()["activation_a0"])


def test_recompute_lowers_a0_for_content_with_close_neighbours() -> None:
    conn = _conn()
    tag = uuid.uuid4().hex[:8]
    base = datetime.now(timezone.utc) - timedelta(days=400)      # 既存データと混ざらない過去
    dim = 1024

    # 先行8件：ほぼ同一方向（互いに近い）。K=7 の近傍が揃う。
    v = np.zeros(dim, dtype=np.float32); v[0] = 1.0
    for i in range(8):
        w = v.copy(); w[1] = 0.01 * i
        w /= np.linalg.norm(w)
        _plant(conn, f"{tag} 似た記録 {i}", w.tolist(), base + timedelta(minutes=i), 1.0)

    # 近い記録（近傍が密）と、離れた記録（孤立）。どちらも a0=1.0 を仮置きしておく。
    near = v.copy(); near[1] = 0.02
    near /= np.linalg.norm(near)
    near_id = _plant(conn, f"{tag} さらに似た記録", near.tolist(), base + timedelta(minutes=20), 1.0)
    far = np.zeros(dim, dtype=np.float32); far[500] = 1.0
    far_id = _plant(conn, f"{tag} まったく別の記録", far.tolist(), base + timedelta(minutes=21), 1.0)

    _run_migration(conn)

    near_a0, far_a0 = _a0(conn, near_id), _a0(conn, far_id)
    conn.close()
    assert near_a0 < 0.3, f"近傍が密な記録の a0 は下がるはず: {near_a0}"
    assert far_a0 > 1.0, f"孤立した記録の a0 は高いまま: {far_a0}"
