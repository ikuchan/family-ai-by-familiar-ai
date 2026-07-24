"""拡散想起 (B) エンティティ辺の DB 読み（dumb・LLM フリー）。

person 中心の再想起＝その人が主体（subject_id）または参加者（participants_json）である
現行版の観測を、新しい順に取る。採点や想起判断は持たない（拡散結線は後続スライス）。
"""

from __future__ import annotations


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
