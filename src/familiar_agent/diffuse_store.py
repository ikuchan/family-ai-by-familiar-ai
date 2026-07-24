"""拡散想起 (B) エンティティ辺の DB 読み（dumb・LLM フリー）。

person 中心の再想起＝その人が主体（subject_id）または参加者（participants_json）である
現行版の観測を、新しい順に取る。採点や想起判断は持たない（拡散結線は後続スライス）。
"""

from __future__ import annotations

import json


def fetch_perspectives(conn, ids: "list[str]") -> "list[dict]":
    """観測 id 群の視点列（subject_id/participants/writer_id）を取る（(B) の種抽出用）。"""
    if not ids:
        return []
    out: list[dict] = []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT subject_id, participants_json, writer_id FROM observations "
            "WHERE id::text = ANY(%s)",
            ([str(i) for i in ids],),
        )
        for subj, part_json, writer in cur.fetchall():
            try:
                participants = json.loads(part_json) if part_json else []
            except (ValueError, TypeError):
                participants = []
            out.append({"subject_id": subj, "participants": participants, "writer_id": writer})
    return out


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
                "score": 0.0, "activation": 0.0, "retrieval_method": "diffuse",
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
    """person が subject または participant の観測 id を新しい順に返す（現行版のみ）。"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM observations "
            "WHERE (subject_id = %s OR participants_json::jsonb ? %s) "
            "AND superseded_by IS NULL "
            "ORDER BY timestamp DESC LIMIT %s",
            (person_id, person_id, limit),
        )
        return [row[0] for row in cur.fetchall()]
