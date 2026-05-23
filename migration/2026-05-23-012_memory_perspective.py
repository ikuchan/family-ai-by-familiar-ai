"""Perspective columns: writer_id, subject_id, participants_json, scope."""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        for col, defn in [
            ("writer_id",         "TEXT"),
            ("subject_id",        "TEXT"),
            ("participants_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("scope",             "TEXT NOT NULL DEFAULT 'speaker'"),
        ]:
            cur.execute(f"""
                ALTER TABLE observations
                ADD COLUMN IF NOT EXISTS {col} {defn}
            """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_obs_writer  ON observations(writer_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_obs_subject ON observations(subject_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_obs_scope   ON observations(scope)
        """)
