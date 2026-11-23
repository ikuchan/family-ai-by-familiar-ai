"""定点ごとの前の見え `pose_views` を足す（知-i 段 2・2026-11-24・本人の承認）。

見えの普通（`pose_norms`・034）は 3 秒ごとに育つ移動平均（係数 0.1）なので、置き物が変わっても十数回で
吸収され、「前と比べて何が変わったか」には使えない。前の見えはそれとは別に、定点ごとに 1 枚を持ち、
見比べて「物が変わった」と判定したとき（と、最初に落ち着いた見えを見たとき）だけ置き換える。

- `embedding`：その見えの DINOv2 埋め込み（384 次元・`pose_norms` と同じ）。足切りの距離に使う。
- `image`：その見えの JPEG。前と今を並べて見比べるのに使う（見回りの画像は 1 つのファイルに書き直すので、
  前の画像はここにしか残らない）。
- `seen_at`：前の見えを置いた時刻。`checked_at`：最後に見比べた時刻（同じ定点を続けて見比べない）。

行は定点の数しかない（実機で 5）。索引は主キーだけ。冪等（`IF NOT EXISTS`）。
src を import せず自前完結させる。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pose_views (
                pose_name  text PRIMARY KEY,
                embedding  vector(384) NOT NULL,
                image      bytea NOT NULL,
                seen_at    timestamptz NOT NULL DEFAULT now(),
                checked_at timestamptz
            )
            """
        )
