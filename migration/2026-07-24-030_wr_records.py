"""WRDB：拡散想起の母集合＝ターンの W スナップショット（想起 MI 集合）を記録する。

拡散想起の (A) 共起辺は「過去に一緒に想起された MI」をたどる。その母集合として、
ターンごとの W の要素 MI id 集合を WR（Working-memory Record）として貯める。

共起カウント（現 W と要素が2つ以上重複する過去 WR を引く）を素直に書けるよう、
ヘッダ `wr_records` と要素の別行 `wr_record_items(wr_id, mi_id)` に分ける
（`GROUP BY` で共有数を数えられる）。記録のみで拡散読み出しには未接続＝挙動不変。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS wr_records (
                id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                created_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS wr_record_items (
                wr_id bigint NOT NULL REFERENCES wr_records(id) ON DELETE CASCADE,
                mi_id text   NOT NULL
            )
            """
        )
        # 共起検索は mi_id で他 WR を引くので mi_id に索引。WR 単位の取り出しに wr_id 索引。
        cur.execute("CREATE INDEX IF NOT EXISTS idx_wr_items_mi ON wr_record_items(mi_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_wr_items_wr ON wr_record_items(wr_id)")
