"""Multi-person support: persons table + person_id FKs.

Reserved IDs:
  AGENT_SELF_ID   = '00000000-0000-0000-0000-000000000000'  agent own memories
  DEFAULT_PERSON_ID = '00000000-0000-0000-0000-000000000001'  legacy default person
"""

AGENT_SELF_ID    = "00000000-0000-0000-0000-000000000000"
DEFAULT_PERSON_ID = "00000000-0000-0000-0000-000000000001"


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id                   TEXT PRIMARY KEY,
                name                 TEXT NOT NULL UNIQUE,
                display_name         TEXT NOT NULL DEFAULT '',
                perspective_vec      BYTEA,
                created_at           TEXT NOT NULL,
                updated_at           TEXT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_persons_name ON persons(name)")

        now = "2026-01-01T00:00:00"
        for pid, name, display in [
            (AGENT_SELF_ID,    "__self__", "Agent self"),
            (DEFAULT_PERSON_ID, "default",  "Default Person"),
        ]:
            cur.execute("""
                INSERT INTO persons (id, name, display_name, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (pid, name, display, now, now))

        tables = [
            "observations", "memory_events", "semantic_facts",
            "behavior_policies", "memory_revisions", "episodes",
            "unfinished_business", "relationship_state",
        ]
        for table in tables:
            cur.execute(f"""
                ALTER TABLE {table}
                ADD COLUMN IF NOT EXISTS person_id TEXT
                    NOT NULL DEFAULT '{DEFAULT_PERSON_ID}'
                    REFERENCES persons(id)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table}_person
                ON {table}(person_id)
            """)
