"""Add activation_a0 and activation_n to observations for the (a0, n) activation representation.

activation_a0: initial activation value a0 (DEFAULT 1.0, migrated from importance).
activation_n:  net delta count n from evaluations (DEFAULT 0; no prior source).

importance is not dropped in this step — recall scoring still reads it.
"""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE observations
            ADD COLUMN IF NOT EXISTS activation_a0 REAL NOT NULL DEFAULT 1.0
        """)
        cur.execute("""
            ALTER TABLE observations
            ADD COLUMN IF NOT EXISTS activation_n INTEGER NOT NULL DEFAULT 0
        """)
        cur.execute("""
            UPDATE observations SET importance = 1.0 WHERE importance IS NULL
        """)
        # importance までは既定値1.0から時間減衰で目減りしただけの値であり、
        # 多少の精度低下を許してそのまま初期値へ簡易移行する。
        cur.execute("""
            UPDATE observations SET activation_a0 = importance
        """)
