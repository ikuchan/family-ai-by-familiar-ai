"""Episodes, activation, unfinished business."""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id                    TEXT PRIMARY KEY,
                title                 TEXT NOT NULL,
                summary               TEXT NOT NULL DEFAULT '',
                participants          TEXT NOT NULL DEFAULT '',
                status                TEXT NOT NULL DEFAULT 'open',
                opened_from_memory_id TEXT,
                created_at            TEXT NOT NULL,
                updated_at            TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS episode_memories (
                id         TEXT PRIMARY KEY,
                episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
                memory_id  TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
                position   INTEGER NOT NULL DEFAULT 0,
                added_at   TEXT NOT NULL,
                UNIQUE(episode_id, memory_id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memory_activation (
                id           TEXT PRIMARY KEY,
                memory_id    TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
                activation   REAL NOT NULL DEFAULT 0.0,
                source       TEXT NOT NULL DEFAULT 'recall',
                context      TEXT NOT NULL DEFAULT '',
                episode_id   TEXT REFERENCES episodes(id) ON DELETE SET NULL,
                activated_at TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS unfinished_business (
                id                TEXT PRIMARY KEY,
                summary           TEXT NOT NULL,
                status            TEXT NOT NULL DEFAULT 'open',
                source            TEXT NOT NULL DEFAULT 'agent',
                related_memory_id TEXT REFERENCES observations(id) ON DELETE SET NULL,
                metadata_json     TEXT NOT NULL DEFAULT '{}',
                created_at        TEXT NOT NULL,
                resolved_at       TEXT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_em_episode ON episode_memories(episode_id, position)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ma_recent  ON memory_activation(activated_at DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ub_status  ON unfinished_business(status, created_at DESC)")
