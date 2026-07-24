"""WRDB（拡散想起の母集合）への dumb な書き込み・読み出し。

ターンの W スナップショット（想起 MI id 集合）を WR として記録する。採点や想起判断は
持たない（[D-WR拡散想起]・拡散読み出しは後続スライス）。
"""

from __future__ import annotations


def save_wr(conn, mi_ids: "list[str]") -> "int | None":
    """ターンの想起 MI id 集合を1つの WR として記録し、wr_id を返す。空なら記録せず None。"""
    ids = [str(m) for m in mi_ids if m]
    if not ids:
        return None
    with conn.cursor() as cur:
        cur.execute("INSERT INTO wr_records DEFAULT VALUES RETURNING id")
        wr_id = int(cur.fetchone()[0])
        cur.executemany(
            "INSERT INTO wr_record_items (wr_id, mi_id) VALUES (%s, %s)",
            [(wr_id, mid) for mid in ids],
        )
    return wr_id


def load_wr_items(conn, wr_id: int) -> "list[str]":
    """WR の要素 MI id を返す（テスト・確認用）。"""
    with conn.cursor() as cur:
        cur.execute("SELECT mi_id FROM wr_record_items WHERE wr_id = %s ORDER BY mi_id", (wr_id,))
        return [row[0] for row in cur.fetchall()]
