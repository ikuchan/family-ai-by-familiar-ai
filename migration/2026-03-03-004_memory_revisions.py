"""Revision history."""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memory_revisions (
                id                  TEXT PRIMARY KEY,
                entity_type         TEXT NOT NULL,
                entity_key          TEXT NOT NULL,
                previous_text       TEXT NOT NULL,
                new_text            TEXT NOT NULL,
                previous_confidence REAL NOT NULL DEFAULT 0.0,
                new_confidence      REAL NOT NULL DEFAULT 0.0,
                source_memory_id    TEXT REFERENCES observations(id) ON DELETE SET NULL,
                reason              TEXT NOT NULL DEFAULT 'projection_update',
                created_at          TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_mr_entity
            ON memory_revisions(entity_type, entity_key, created_at)
        """)
