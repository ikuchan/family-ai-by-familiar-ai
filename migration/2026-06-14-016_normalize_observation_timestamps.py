"""Normalize observations timestamp: TEXT → TIMESTAMPTZ, drop redundant date/time columns.

Precondition: all existing timestamp values must be ISO 8601 strings (YYYY-MM-DD…).
The pre-check raises RuntimeError if any non-conforming row is found.
"""


def _check_timestamps(conn) -> None:
    """Raise RuntimeError if any observation has a non-ISO 8601 timestamp."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) AS cnt FROM observations
            WHERE timestamp IS NOT NULL
              AND timestamp !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}'
            """
        )
        row = cur.fetchone()
        bad = row[0] if isinstance(row, tuple) else row["cnt"]
    if bad:
        raise RuntimeError(
            f"Cannot migrate: {bad} observation(s) have non-ISO timestamps. "
            "Fix them manually before running migration 016."
        )


def upgrade(conn) -> None:
    _check_timestamps(conn)
    with conn.cursor() as cur:
        cur.execute(
            """
            ALTER TABLE observations
            ALTER COLUMN timestamp TYPE TIMESTAMPTZ
            USING timestamp::timestamptz
            """
        )
        cur.execute("ALTER TABLE observations DROP COLUMN IF EXISTS date")
        cur.execute("ALTER TABLE observations DROP COLUMN IF EXISTS time")
