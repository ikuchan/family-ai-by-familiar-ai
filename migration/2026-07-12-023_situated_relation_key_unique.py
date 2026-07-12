"""situated V2 スライス2：UNIQUE を (obs_id, person_id, relation_key) へ付け替え。

[D-在席相関/V2]：同一 (obs_id, person_id) に relation_key の違う複数の関係行を許す。
旧 UNIQUE(obs_id, person_id)（制約名 situated_embeddings_obs_id_person_id_key）を落とし、
UNIQUE(obs_id, person_id, relation_key) を張る。生成はまだ 'presence' のみなので
実際に並ぶ行数は変わらず挙動は不変。ON CONFLICT の同定キーはコード側で揃える。
"""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE situated_embeddings
            DROP CONSTRAINT IF EXISTS situated_embeddings_obs_id_person_id_key
        """)
        cur.execute("""
            ALTER TABLE situated_embeddings
            DROP CONSTRAINT IF EXISTS situated_embeddings_obs_person_relation_key
        """)
        cur.execute("""
            ALTER TABLE situated_embeddings
            ADD CONSTRAINT situated_embeddings_obs_person_relation_key
            UNIQUE (obs_id, person_id, relation_key)
        """)
