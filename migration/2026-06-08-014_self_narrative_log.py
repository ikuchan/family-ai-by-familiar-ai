"""Self-narrative session diary log."""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS self_narrative_log (
                id         BIGSERIAL PRIMARY KEY,
                date       TEXT NOT NULL,
                text       TEXT NOT NULL,
                mood       TEXT NOT NULL DEFAULT 'neutral',
                trigger    TEXT NOT NULL DEFAULT 'session_close',
                created_at TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS self_narrative_log_date_idx
            ON self_narrative_log (date)
        """)
