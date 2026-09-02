"""拡散想起 (B) エンティティ辺の DB 読み（dumb・LLM フリー）。

person 中心の再想起＝その人が**話題の主体**（`about`）か**そばに居た**（`present`）面を
持つ現行版の観測を、新しい順に取る。採点や想起判断は持たない。

段4 で視点列から situated の面へ移した。**母集合に `actor` は入れない。** その人が
「やった」だけの記録まで入れると、パジュ自身が書いた記録（`actor` が `__self__` の
6433 行）がどの種からも湧く。種として使うのと、母集合にするのは別の問いである。
"""

from __future__ import annotations

import numpy as np

from .store.embedding import _decode_vector


def order_ids_by_farthest(conn, ids: "list[str]", seed_vec) -> "list[str]":
    """候補 id を seed ベクトルから遠い順（コサイン低い順＝新規性高い順）に並べ替える（4b）。

    埋め込みが無い/次元不一致の候補は末尾へ。DB 非破壊・LLM フリー。
    """
    ids = [str(i) for i in ids if i]
    if not ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT obs_id, vector FROM obs_embeddings WHERE obs_id::text = ANY(%s)",
            (ids,),
        )
        rows = cur.fetchall()
    s = np.asarray(seed_vec, dtype=np.float32)
    s = s / (float(np.linalg.norm(s)) or 1.0)
    cos: dict[str, float] = {}
    for oid, blob in rows:
        if blob is None:
            continue
        v = _decode_vector(bytes(blob))
        if v.size != s.size:
            continue
        v = v / (float(np.linalg.norm(v)) or 1.0)
        cos[str(oid)] = float(np.dot(s, v))
    # 遠い順＝コサイン昇順。埋め込み無しは末尾（新規性不明）。
    return sorted(ids, key=lambda i: (i not in cos, cos.get(i, 1.0)))


def fetch_relation_persons(conn, ids: "list[str]") -> "list[dict]":
    """観測 id 群の関係の面（誰が・どの役割で）を取る（(B) の種抽出用）。

    返すのは `person_id` と `relation_key` の行で、並べ替えは `select_entity_seeds` が
    役割の優先順で行う。ここは dumb な読み出しに徹する。
    """
    if not ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT person_id, relation_key FROM situated_memories "
            "WHERE obs_id = ANY(%s)",
            ([str(i) for i in ids],),
        )
        return [{"person_id": r[0], "relation_key": r[1]} for r in cur.fetchall()]


def fetch_diffuse_rows(conn, ids: "list[str]") -> "list[dict]":
    """拡散で足す MI の最小 W 要素（memory dict）を取る。a0=0（重み0）で W へ付す。"""
    if not ids:
        return []
    rows: list[dict] = []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, content, timestamp, direction, kind, emotion FROM observations "
            "WHERE id::text = ANY(%s)",
            ([str(i) for i in ids],),
        )
        by_id = {
            str(r[0]): {
                "memory_id": str(r[0]), "summary": r[1], "timestamp": r[2],
                "direction": r[3], "kind": r[4], "source_kind": r[4], "emotion": r[5],
                "fit": 0.0, "groundedness": 0.0, "retrieval_method": "diffuse",
            }
            for r in cur.fetchall()
        }
    # 呼び出し順（追加順）を保つ。
    for i in ids:
        row = by_id.get(str(i))
        if row is not None:
            rows.append(row)
    return rows


def cooccurring_mi_ids(
    conn, w_mi_ids: "list[str]", min_shared: int = 2, limit: int = 20
) -> "list[str]":
    """(A) 共起辺：現 W と要素が min_shared 件以上重複する過去 WR の要素 MI を候補で返す。

    現 W 自身の id と自己認識 MI（kind='self_model'）は除く（想起MIリスト・[D-WR拡散想起]）。
    「seed 最遠の選別（新規性）」はしない（後続スライスで (B) と共通に絞る）。
    """
    ids = [str(m) for m in w_mi_ids if m]
    if len(ids) < min_shared:
        return []
    with conn.cursor() as cur:
        cur.execute(
            "WITH matched AS ("
            "  SELECT wr_id FROM wr_record_items WHERE mi_id = ANY(%s)"
            "  GROUP BY wr_id HAVING count(DISTINCT mi_id) >= %s"
            ") "
            "SELECT DISTINCT i.mi_id FROM wr_record_items i "
            "JOIN matched m ON i.wr_id = m.wr_id "
            "JOIN observations o ON o.id::text = i.mi_id AND o.kind <> 'self_model' "
            "WHERE NOT (i.mi_id = ANY(%s)) "
            "LIMIT %s",
            (ids, min_shared, ids, limit),
        )
        return [row[0] for row in cur.fetchall()]


def recall_by_person(conn, person_id: str, limit: int = 5) -> "list[str]":
    """その person が `about` か `present` の面を持つ観測 id を新しい順に返す（現行版のみ）。

    同じ観測に両方の面が立つことがあるので、観測ごとに畳んで返す。
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT s.obs_id FROM situated_memories s "
            "JOIN observations o ON o.id = s.obs_id AND o.superseded_by IS NULL "
            "WHERE s.person_id = %s AND s.relation_key IN (\'about\', \'present\') "
            "GROUP BY s.obs_id, o.timestamp "
            "ORDER BY o.timestamp DESC LIMIT %s",
            (person_id, limit),
        )
        return [row[0] for row in cur.fetchall()]


def shared_memory_ids(
    conn, person_ids: "list[str]", limit: int = 20
) -> "list[str]":
    """在席者**全員**が関係を持つ観測 id を新しい順に返す（共通の記憶・段4）。

    その場に居合わせた人たちで共有している出来事である。片方としか関係の無い観測は
    入れない。在席者が一人なら「共通の」記憶は無いので空を返す。

    役割は問わない。誰かが話題で誰かが居ただけ、という混ざり方も共有には違いない。
    """
    pids = list(dict.fromkeys(str(p) for p in person_ids if p))
    if len(pids) < 2:
        return []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT s.obs_id FROM situated_memories s "
            "JOIN observations o ON o.id = s.obs_id AND o.superseded_by IS NULL "
            "WHERE s.person_id = ANY(%s) "
            "GROUP BY s.obs_id, o.timestamp "
            "HAVING count(DISTINCT s.person_id) = %s "
            "ORDER BY o.timestamp DESC LIMIT %s",
            (pids, len(pids), limit),
        )
        return [row[0] for row in cur.fetchall()]
