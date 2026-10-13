"""意図の台帳の記録（出-cc・2026-10-13・本人の決定）。

意図（調べもの・道具・主LLM の答え）ごとに uuid を振り、完了は uuid で引く（`loop/intent_ledger`）。引き当てはメモリ上の
台帳で行い、ここはあとから追うための記録：投げた時刻、どう終わったか（結果が届いた・打ち切った・求めと一緒に閉じた・
閉じた後に届いた）、失敗の印、結果の字数。結果の中身は持たない（O にあり、家族の話も含む）。消さない。冪等。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS intents (
                id           uuid PRIMARY KEY,
                request_id   text NOT NULL DEFAULT '',
                origin       text NOT NULL DEFAULT '',
                action       text NOT NULL,
                query        text NOT NULL DEFAULT '',
                state        text NOT NULL,
                failed       boolean NOT NULL DEFAULT false,
                result_chars integer,
                opened_at    timestamptz NOT NULL DEFAULT now(),
                ended_at     timestamptz
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_intents_request_id ON intents(request_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_intents_opened_at ON intents(opened_at)")
