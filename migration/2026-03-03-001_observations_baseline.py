"""Baseline observations schema (PostgreSQL)."""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id            TEXT PRIMARY KEY,
                content       TEXT NOT NULL,
                timestamp     TEXT NOT NULL,
                date          TEXT NOT NULL,
                time          TEXT NOT NULL,
                direction     TEXT NOT NULL DEFAULT 'unknown',
                kind          TEXT NOT NULL DEFAULT 'observation',
                emotion       TEXT NOT NULL DEFAULT 'neutral',
                image_path    TEXT,
                image_data    TEXT,
                importance    REAL NOT NULL DEFAULT 1.0,
                superseded_by TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS obs_embeddings (
                obs_id  TEXT PRIMARY KEY REFERENCES observations(id) ON DELETE CASCADE,
                vector  BYTEA NOT NULL
            )
        """)
        for sql in [
            "CREATE INDEX IF NOT EXISTS idx_obs_timestamp  ON observations(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_obs_date       ON observations(date)",
            "CREATE INDEX IF NOT EXISTS idx_obs_kind       ON observations(kind)",
            "CREATE INDEX IF NOT EXISTS idx_obs_superseded ON observations(superseded_by)",
        ]:
            cur.execute(sql)
