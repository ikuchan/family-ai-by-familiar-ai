"""Mental state append-only log."""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS mental_state_log (
                id         BIGSERIAL PRIMARY KEY,
                turn_index INTEGER NOT NULL DEFAULT 0,
                created_at TEXT    NOT NULL,
                payload    TEXT    NOT NULL
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS mental_state_log_created_at_idx
            ON mental_state_log (created_at)
        """)
