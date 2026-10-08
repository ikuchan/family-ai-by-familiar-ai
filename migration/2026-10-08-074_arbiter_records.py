"""調停ごとの記録（出-ay 段 3・2026-10-08・本人の決定）。

正解（`JEV_正解.md`・版管理の外）と突き合わせて Jev のプロンプトを直すため、調停のたびに Jev に渡したそのままの文
（`state`・W を含む）・問い（`questions`）・答え（`answer`）・結末（`outcome`）を残す。W は DEBUG のログにしか無く、
後から組み直すと想起の点数が変わって同じにならない。記-o（W に何が載っていたか）もこれで見る。消さない（本人の決定ア）。
冪等。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS arbiter_records (
                id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                at        timestamptz NOT NULL DEFAULT now(),
                origin    text NOT NULL,
                utterance text NOT NULL DEFAULT '',
                state     text NOT NULL,
                questions jsonb NOT NULL,
                answer    jsonb NOT NULL,
                outcome   text NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_arbiter_records_at ON arbiter_records(at)")
