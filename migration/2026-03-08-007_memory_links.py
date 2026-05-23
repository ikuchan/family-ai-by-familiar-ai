"""Typed associative links."""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memory_links (
                id         TEXT PRIMARY KEY,
                source_id  TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
                target_id  TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
                link_type  TEXT NOT NULL DEFAULT 'related',
                note       TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(source_id, target_id, link_type)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ml_source ON memory_links(source_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ml_target ON memory_links(target_id)")
