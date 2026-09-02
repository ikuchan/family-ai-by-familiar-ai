"""視点ベクトルと situated 行の作成・保存。

`situated_memories` は記憶を関係の面ごとに持つ表で、その面のベクトルが想起の
母集合を作る（[D-在席相関/V2]）。ベクトルは「観測 × 関係」で決まり人には依らない。
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
from ..person_memory_manager import AGENT_SELF_ID, DEFAULT_PERSON_ID
from .embedding import (
    EMBEDDING_DIM,
    _coerce_to_embedding_dim,
    _decode_vector,
    _normalise,
)

from .context import StoreContext

logger = logging.getLogger(__name__)


def _situated_vector(
    mem_vec: "np.ndarray", mu: "np.ndarray | None",
) -> "np.ndarray":
    """situated ベクトルを作る（平均中心化 C2）。

    コサインを取る前に共通成分 mu を引いて L2 正規化する（計測台帳 §1）。生コサインは
    異方性（cone 効果）で無関係でも 0.88 付近に圧縮され、関連との窓が 0.016 しかない。
    mu を引くと無関係が ≈0 へ移り窓が約12倍になる。

    **書き込み（situated 生成）と問い合わせ（recall のクエリ）は必ずこの同じ関数を通す**。
    片方だけ中心化すると別空間になりコサインが無意味になるため、式を1箇所に集約する。
    mu が None（未推定・次元不一致）なら中心化せず素のベクトルを正規化して返す。

    **人の視点は入らない**（045）。かつては視点ベクトルを係数つきで足して人ごとに
    寄せていたが、
    差は 047 が足す関係項が担う。ベクトルは「観測 × 関係」だけで決まり、面の言葉
    （役割の接頭辞＋出来事の本文）から作るので誰かは入らない（[D-在席相関/V2]）。
    """
    composed = mem_vec - mu if mu is not None else mem_vec
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

    def situate(self, mem_vec: np.ndarray, conn) -> np.ndarray:
        """内容ベクトルを situated 化する（recall のクエリと同じ式）。

        conn を渡すのは、書き込み経路（ロック保持中）から呼ぶため（mu の読みでロックを
        取り直さない）。novelty 算出などで内容の situated クエリを作るのに使う。
        **人は取らない**（045 で視点項が消えた）。
        """
        return _situated_vector(
            _coerce_to_embedding_dim(np.asarray(mem_vec, dtype=np.float32)),
            self._embedding_mu(conn),
        )


    def reembed_facets(self, conn, obs_id: str, mem_vec: np.ndarray, body: str) -> None:
        """本文が変わったとき、**いま立っている面をなぞって**ベクトルを作り直す（段5）。

        面を作り直さないのは、**面が正**だからである。誰との関係かは面が持っており、
        観測の側にはもう残っていない。作り直そうとすると材料が無い。

        **REST が足した意味役割の面も、ここでベクトルが新しくなる。** 以前は `actor` と
        `present` を作り直すだけだったので、`about` などの面は古い本文のベクトルを持った
        まま取り残されていた。想起はベクトルで探すので、取り残されると本文と食い違う。

        **言葉は機械が立てた面だけ書き直す。** `present` の `[そばに居た] ` ＋ 本文は機械が
        作ったものなので本文に追随させるが、REST が本文を読んで書いた言葉は REST のもので、
        機械が上書きしてよいものではない（段①と段②の切り分け・047）。
        """
        situated = _situated_vector(
            _coerce_to_embedding_dim(mem_vec), self._embedding_mu(conn)
        )
        vec_str = vec_to_sql(situated.tolist())
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE situated_memories SET vector = %s::vector, "
                "  content = CASE WHEN relation_key = %s THEN %s ELSE content END "
                "WHERE obs_id = %s",
                (vec_str, "present", f"[そばに居た] {body}", obs_id),
            )

    def _upsert_situated_embedding(
        self,
        conn,
        obs_id: str,
        person_id: str,
        mem_vec: np.ndarray,
        relation_key: str = "present",
        content: "str | None" = None,
    ) -> None:
        """一つの面（obs × person × relation）のベクトルと言葉を書く。

        relation_key は関係の帳簿ラベル（[D-在席相関/V2]）。同定キーは
        (obs_id, person_id, relation_key) で、同じ関係の再計算は vector を更新する。

        `content` は**その面から見た言葉**＝`[役割の札] ` ＋ 出来事の本文。
        `actor` だけは持たない（全観測に立つので書き直す意味がない）。
        """
        mem_vec = _coerce_to_embedding_dim(mem_vec)
        # 書き込み経路は db.lock を保持したまま来るので conn を渡す（再入不可のため）。
        situated = _situated_vector(mem_vec, self._embedding_mu(conn))
        vec_str = vec_to_sql(situated.tolist())
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO situated_memories "
                "(id, obs_id, person_id, vector, relation_key, content) "
                "VALUES (%s, %s, %s, %s::vector, %s, %s) "
                "ON CONFLICT (obs_id, person_id, relation_key) DO UPDATE SET "
                "  vector = EXCLUDED.vector, content = EXCLUDED.content",
                (str(uuid.uuid4()), obs_id, person_id, vec_str, relation_key, content),
            )

    def refresh_situated_memories(
        self,
        conn,
        obs_id: str,
        mem_vec: np.ndarray,
        *,
        body: str,
        writer_id: str,
        participants: "list[str] | None" = None,
    ) -> None:
        """その観測に**関係のある人**の面だけを立てる（047）。

        **面の生成は二段である。** ここが担うのは段①＝機械で確実に出るものだけ。

          `actor`（誰がやったか）  ← `writer_id`。観測1件につき必ず1行。content 無し
          `present`（誰が居たか）  ← `participants_json` の各在席者。
                                     content は `[そばに居た] ` ＋ 出来事の本文

        段②（`addressee`／`about`／`experiencer`／`beneficiary`／`companion`／
        `source`／`owner` …）は **REST 内省が本文を読んで足す**。既存の観測にも
        さかのぼって足していく（[D-在席相関/V2]「relation_key 語彙は REST が育て・畳む」）。

        **047 の前は登録人物全員＋AGENT_SELF に行を作っていた**（観測1件につき人数ぶん）。
        全員に同じベクトルが入るので、その人がその記憶とどう関わったかを表していなかった。

        `writer_id` が `default`（話者未解決）のときは `actor` を `__self__` にする
        （048「内なる記録はエージェントのもの」）。話者が解決できなかった記録は、
        パジュ自身がしたことだからである。

        **材料は引数で受け取る**（段5）。以前は `obs_id` から観測を読み直していたが、
        `writer_id` と `participants_json` の列を落としたので読む先が無い。誰がしたこと・
        誰が居たかは書き込みの瞬間に分かっているので、そのまま渡す。立った面が以後の正で、
        観測の行に残しておく必要はない。
        """
        writer = str(writer_id or DEFAULT_PERSON_ID)
        # 048：話者が解決できなかった記録は、パジュ自身がしたこと。
        actor = AGENT_SELF_ID if writer == DEFAULT_PERSON_ID else writer
        participants = list(participants or [])

        wanted: list[tuple[str, str, str | None]] = [(actor, "actor", None)]
        for pid in participants:
            if not pid:
                continue
            wanted.append((str(pid), "present", f"[そばに居た] {body}"))

        for pid, key, content in wanted:
            self._upsert_situated_embedding(conn, obs_id, pid, mem_vec,
                                            relation_key=key, content=content)

        # 関係の無くなった面は落とす。**段②（REST が足した意味役割）は残す**ので、
        # 対象は機械が立てる2種（`actor`／`present`）に限る。
        keep_p = [p for p, _, _ in wanted]
        keep_r = [k for _, k, _ in wanted]
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM situated_memories sm "
                "WHERE sm.obs_id = %s AND sm.relation_key IN ('actor', 'present') "
                "  AND NOT EXISTS ("
                "    SELECT 1 FROM unnest(%s::text[], %s::text[]) AS k(p, r) "
                "     WHERE k.p = sm.person_id AND k.r = sm.relation_key)",
                (obs_id, keep_p, keep_r),
            )
