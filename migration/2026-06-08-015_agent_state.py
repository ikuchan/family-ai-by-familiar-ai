"""Generic agent state key-value store (desires, self_state, heartbeat, concerns, etc.)."""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS agent_state (
                state_key   TEXT PRIMARY KEY,
                value_json  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)
