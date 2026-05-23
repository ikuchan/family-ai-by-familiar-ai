"""Semantic facts and behavior policies."""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS semantic_facts (
                id               TEXT PRIMARY KEY,
                fact_key         TEXT NOT NULL UNIQUE,
                fact_text        TEXT NOT NULL,
                source_memory_id TEXT REFERENCES observations(id) ON DELETE SET NULL,
                confidence       REAL NOT NULL DEFAULT 0.5,
                tags             TEXT NOT NULL DEFAULT '',
                last_seen_at     TEXT NOT NULL,
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sf_last_seen ON semantic_facts(last_seen_at)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS behavior_policies (
                id               TEXT PRIMARY KEY,
                policy_key       TEXT NOT NULL UNIQUE,
                policy_text      TEXT NOT NULL,
                trigger_context  TEXT NOT NULL DEFAULT '',
                action_hint      TEXT NOT NULL DEFAULT '',
                source_memory_id TEXT REFERENCES observations(id) ON DELETE SET NULL,
                confidence       REAL NOT NULL DEFAULT 0.5,
                last_seen_at     TEXT NOT NULL,
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_bp_last_seen ON behavior_policies(last_seen_at)")
