"""視点ベクトルと situated 行の作成・保存。

`situated_memories` は記憶を「誰の視点から見たか」に寄せたベクトルで、想起の
母集合を作る（[D-在席相関/V2]）。人ごとの視点ベクトル `perspective_vec` と、
中心化に使う埋め込みの平均 mu（`embedding_means`）もここに属する。

**ベクトルの作成と保存だけを持つ。**想起（W の構築）は持たない。書き込みと
問い合わせで別の式を使うと、同じベクトルが別空間に散る事故が起きるため、合成式は
`_situated_vector` の1本に限る（2026-07-18 の平均中心化 C2 でこの形にした）。
"""

from __future__ import annotations

import logging
import uuid

import numpy as np

from ..db import get_db, vec_to_sql
from ..person_memory_manager import AGENT_SELF_ID, ALPHA
from .embedding import (
    EMBEDDING_DIM,
    _coerce_to_embedding_dim,
    _decode_vector,
    _encode_vector,
    _normalise,
)

from . import clock
from .context import StoreContext

logger = logging.getLogger(__name__)


def _situated_vector(
    mem_vec: "np.ndarray", p_vec: "np.ndarray", mu: "np.ndarray | None",
) -> "np.ndarray":
    """situated ベクトルを作る（平均中心化 C2）。

    コサインを取る前に共通成分 mu を引いて L2 正規化する（計測台帳 §1）。生コサインは
    異方性（cone 効果）で無関係でも 0.88 付近に圧縮され、関連との窓が 0.016 しかない。
    mu を引くと無関係が ≈0 へ移り窓が約12倍になる。

    **書き込み（situated 生成）と問い合わせ（recall のクエリ）は必ずこの同じ関数を通す**。
    片方だけ中心化すると別空間になりコサインが無意味になるため、式を1箇所に集約する。
    mu が None（未推定・次元不一致）なら中心化せず従来式で返す（フォールバック）。
    """
    composed = mem_vec + ALPHA * p_vec
    if mu is not None:
        composed = composed - mu
    return _normalise(composed)


def load_embedding_mean(dim: int, conn=None) -> "np.ndarray | None":
    """埋め込みの平均ベクトル mu（global）を読む（平均中心化）。

    コサインを取る前に共通成分（cone）を除くために引くベクトル（計測台帳 §1）。
    行が無い、または保存された次元が `dim` と一致しない（埋め込みモデルを替えた後など）
    ときは None を返し、呼び出し側は**中心化しない**でフォールバックする。

    `conn` を渡すとその接続で読む。`db.lock` は**再入不可**で、書き込み経路
    （`_materialize_save_event`）はロックを保持したまま situated 生成を呼ぶため、
    そこから来るときは必ず `conn` を渡してロックを取り直さない（二重取得でデッドロックする）。
    """
    if conn is not None:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT dim, vector FROM embedding_means "
                "WHERE scope = %s AND scope_key = %s",
                ("global", ""),
            )
            row = cur.fetchone()
    else:
        db = get_db()
        with db.lock:
            conn2 = db.conn()
            with conn2.cursor() as cur:
                cur.execute(
                    "SELECT dim, vector FROM embedding_means "
                    "WHERE scope = %s AND scope_key = %s",
                    ("global", ""),
                )
                row = cur.fetchone()
    if not row:
        return None
    stored_dim, blob = (row[0], row[1]) if not isinstance(row, dict) else (row["dim"], row["vector"])
    if stored_dim != dim or blob is None:
        return None
    return _decode_vector(bytes(blob))


class SituatedVectors:
    """視点ベクトルと situated 行の持ち主。

    使うものは文脈（`StoreContext`）から受け取る。宿主の名前空間は覗かない。
    """

    def __init__(self, ctx: StoreContext) -> None:
        self._ctx = ctx

    def _embedding_mu(self, conn=None) -> "np.ndarray | None":
        """平均中心化に使う mu を返す（C2・遅延読み込みで1回だけ）。

        mu は固定値（低頻度で再推定）なので、書き込みと想起のたびに DB を引かずに
        インスタンスへ持つ。再推定時の無効化は REST 接続の段で扱う。未推定・次元不一致は
        None で、そのとき `_situated_vector` は中心化しない。

        `db.lock` は再入不可なので、**ロックを保持したまま呼ぶ経路（書き込み）は `conn` を
        渡す**。渡さないと同じロックを二重取得してデッドロックする。
        """
        if not hasattr(self, "_mu_cache"):
            self._mu_cache = load_embedding_mean(EMBEDDING_DIM, conn)
        return self._mu_cache

    def situate(self, mem_vec: np.ndarray, person_id: str, conn) -> np.ndarray:
        """内容ベクトルを person 視点で situated 化する（recall のクエリと同じ式）。

        conn を渡すのは、書き込み経路（ロック保持中）から呼ぶため（p_vec・mu の読みで
        ロックを取り直さない）。novelty 算出などで内容の situated クエリを作るのに使う。
        """
        p_vec = self._get_perspective_vec_with_conn(person_id, conn)
        mu = self._embedding_mu(conn)
        return _situated_vector(
            _coerce_to_embedding_dim(np.asarray(mem_vec, dtype=np.float32)), p_vec, mu
        )

    def _get_perspective_vec_with_conn(self, person_id: str, conn) -> np.ndarray:
        """Load perspective vector using an already-open connection (no lock)."""
        with conn.cursor() as cur:
            cur.execute("SELECT perspective_vec FROM persons WHERE id = %s", (person_id,))
            row = cur.fetchone()
        if row and row["perspective_vec"]:
            return _coerce_to_embedding_dim(_decode_vector(bytes(row["perspective_vec"])))
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    def _get_perspective_vec(self, person_id: str) -> np.ndarray:
        """Load person's perspective vector from DB. Returns zeros if none."""
        with self._ctx.lock:
            conn = self._ctx.conn()
            return self._get_perspective_vec_with_conn(person_id, conn)

    def update_perspective_vec(self, person_id: str, mem_vec: np.ndarray, lr: float = 0.05) -> None:
        """Moving-average update of person's perspective vector."""
        mem_vec = _coerce_to_embedding_dim(mem_vec)
        old = self._get_perspective_vec(person_id)
        new = _normalise((1.0 - lr) * old + lr * mem_vec)
        blob = _encode_vector(new.tolist())
        with self._ctx.lock:
            conn = self._ctx.conn()
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE persons SET perspective_vec = %s, updated_at = %s WHERE id = %s",
                    (blob, clock.now_utc_iso(), person_id),
                )
            conn.commit()

    def _upsert_situated_embedding(
        self,
        conn,
        obs_id: str,
        person_id: str,
        mem_vec: np.ndarray,
        relation_key: str = "presence",
    ) -> None:
        """Compute and store situated vector for one person under a relation_key.

        relation_key は関係の帳簿ラベル（[D-在席相関/V2]）。同定キーは
        (obs_id, person_id, relation_key) で、同じ関係の再計算は vector を更新する。
        生成の多型化（speaker/subject）は後続スライス。
        """
        mem_vec = _coerce_to_embedding_dim(mem_vec)
        p_vec = self._get_perspective_vec_with_conn(person_id, conn)
        # 書き込み経路は db.lock を保持したまま来るので conn を渡す（再入不可のため）。
        situated = _situated_vector(mem_vec, p_vec, self._embedding_mu(conn))
        vec_str = vec_to_sql(situated.tolist())
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO situated_memories (id, obs_id, person_id, vector, relation_key) "
                "VALUES (%s, %s, %s, %s::vector, %s) "
                "ON CONFLICT (obs_id, person_id, relation_key) DO UPDATE SET vector = EXCLUDED.vector",
                (str(uuid.uuid4()), obs_id, person_id, vec_str, relation_key),
            )

    def refresh_situated_memories(self, conn, obs_id: str, mem_vec: np.ndarray) -> None:
        """Pre-compute situated vectors for ALL registered persons + agent self."""
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM persons")
            person_ids = [row["id"] for row in cur.fetchall()]
        for pid in person_ids:
            self._upsert_situated_embedding(conn, obs_id, pid, mem_vec)
        # Always include AGENT_SELF_ID
        if AGENT_SELF_ID not in person_ids:
            self._upsert_situated_embedding(conn, obs_id, AGENT_SELF_ID, mem_vec)
