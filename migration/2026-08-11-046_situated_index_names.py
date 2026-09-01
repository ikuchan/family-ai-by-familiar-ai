"""索引名と制約名を表の改名（044）へ追随させる。

PostgreSQL の `ALTER TABLE ... RENAME TO ...` は**索引名も制約名も変えない**。044 で
`situated_embeddings` を `situated_memories` へ改めたあと、名前だけが旧いまま残っていた。

**挙動は変わらない。** 索引名は問い合わせ計画に影響しない。復元されたスキーマを
2026-08-21 のダンプと一字一句そろえるための段である。

素の索引は `ALTER INDEX`、制約（主キー・一意・外部キー）は
`ALTER TABLE ... RENAME CONSTRAINT` で、文が違う。制約名を変えると裏の索引名も一緒に
変わるので、主キーと一意制約はここで一度だけ扱う。

`RENAME CONSTRAINT` には `IF EXISTS` が無いので、`pg_constraint` を引いてから分岐する
（既に新しい名前なら何もしない＝冪等）。

`idx_situated_recency` は 044 が正しい名前で作ったので触らない。

**識別子は SQL の引数にできないので f-string で組む。** ただし値は**この表に限った固定の
対応表**からしか来ない（外から受け取った文字列は入らない）。

この id は 2026-08-21 のダンプに残っていたものと同じである。本番の
`schema_migrations` には既に記録されているので、ここでは流れない。
"""

from __future__ import annotations

# 旧名 → 新名。表の改名に追随させるだけで、対象も名前も固定である。
_INDEXES = (
    ("idx_se_person", "idx_situated_person"),
    ("idx_se_hnsw", "idx_situated_hnsw"),
)
_CONSTRAINTS = (
    ("situated_embeddings_pkey", "situated_memories_pkey"),
    ("situated_embeddings_obs_person_relation_key",
     "situated_memories_obs_person_relation_key"),
    ("situated_embeddings_obs_id_fkey", "situated_memories_obs_id_fkey"),
    ("situated_embeddings_person_id_fkey", "situated_memories_person_id_fkey"),
)


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        for old, new in _INDEXES:
            cur.execute(f"ALTER INDEX IF EXISTS {old} RENAME TO {new}")
        for old, new in _CONSTRAINTS:
            cur.execute(
                "SELECT 1 FROM pg_constraint "
                "WHERE conrelid = 'situated_memories'::regclass AND conname = %s",
                (old,),
            )
            if cur.fetchone():
                cur.execute(
                    f"ALTER TABLE situated_memories RENAME CONSTRAINT {old} TO {new}"
                )
