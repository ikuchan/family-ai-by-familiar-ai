"""Add recall_count and last_recalled_at to observations for memory decay scoring.

recall_count:     how many times this memory has been recalled in conversation (DEFAULT 0).
                  Effective half-life = base_half_life * 2^recall_count.
last_recalled_at: when the memory was last surfaced; resets the decay clock.
                  NULL means never recalled → decay starts from observations.timestamp.
"""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE observations
            ADD COLUMN IF NOT EXISTS recall_count INT NOT NULL DEFAULT 0
        """)
        cur.execute("""
            ALTER TABLE observations
            ADD COLUMN IF NOT EXISTS last_recalled_at TIMESTAMPTZ
        """)
