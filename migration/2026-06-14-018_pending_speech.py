"""Add pending_speech table for Issue D: note_to_share / whom_to_address flow.

Each row represents a memory the agent wants to share with someone.
observation_id references observations(id) ON DELETE CASCADE (containment guarantee).
target_person_id is nullable: NULL means "any present person".
reinforce_count is reserved for future strengthening mechanic (Issue D+).
"""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pending_speech (
                id               TEXT PRIMARY KEY,
                observation_id   TEXT NOT NULL
                                 REFERENCES observations(id) ON DELETE CASCADE,
                target_person_id TEXT,
                created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
                reinforce_count  INT NOT NULL DEFAULT 0
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_pending_target "
            "ON pending_speech(target_person_id)"
        )
