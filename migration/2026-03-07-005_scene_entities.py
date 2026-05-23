"""Scene entities, events, exploration state."""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS scene_entities (
                entity_id  TEXT PRIMARY KEY,
                label      TEXT NOT NULL,
                category   TEXT NOT NULL DEFAULT 'object',
                first_seen TEXT NOT NULL,
                last_seen  TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.8,
                bbox_hint  TEXT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_se_label ON scene_entities(label)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS scene_events (
                event_id     TEXT PRIMARY KEY,
                event_type   TEXT NOT NULL,
                entity_id    TEXT REFERENCES scene_entities(entity_id) ON DELETE SET NULL,
                entity_label TEXT NOT NULL,
                timestamp    TEXT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sev_ts ON scene_events(timestamp)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS exploration_state (
                id           INTEGER PRIMARY KEY,
                pan_accum    REAL NOT NULL DEFAULT 0.0,
                tilt_accum   REAL NOT NULL DEFAULT 0.0,
                records_json TEXT NOT NULL DEFAULT '[]',
                saved_at     TEXT NOT NULL,
                CONSTRAINT chk_single_row CHECK (id = 1)
            )
        """)
