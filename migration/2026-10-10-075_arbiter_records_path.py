"""調停の記録に、道・意味・最終の動作を足す（出-ay 段 5d・2026-10-10・本人の決定）。

段 4 で調停を起点ごとの道（発話・完了・情動・機器）に組み替えた。起点の欄（`origin`）だけでは完了か分からない（発話から
始まった求めの完了は「発話」と残る）ので、道（`path`）・発話の意味（`meaning`）・最終の動作（`final`）を足す。いままでの
行は空のまま（古い分岐の問いの記録）。冪等。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        for column in ("path", "meaning", "final"):
            cur.execute(
                f"ALTER TABLE arbiter_records ADD COLUMN IF NOT EXISTS {column} text NOT NULL DEFAULT ''"
            )
