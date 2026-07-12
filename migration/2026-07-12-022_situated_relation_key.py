"""Add relation_key to situated_embeddings — situated V2 型つき関係エッジの器（第一歩）。

[D-在席相関/V2]：situated を「MI×person の型つき関係エッジ」へ精緻化する準備。
本マイグレーションは列の追加だけで、生成・想起の母集合は変えない。

relation_key: 関係の帳簿用ラベル（検索には使わない）。既定 'presence'。
NOT NULL DEFAULT で既存行は 'presence' に埋まる（在席関係とみなす）。

UNIQUE(obs_id, person_id) はこの段では据え置き。複数関係を許す UNIQUE 撤去、
関係エッジの生成（speaker/subject）、person_id 削除は後続スライスで行う。
"""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE situated_embeddings
            ADD COLUMN IF NOT EXISTS relation_key TEXT NOT NULL DEFAULT 'presence'
        """)
