"""Situated (person-perspective) embeddings using pgvector.

Pre-computed at write time: situated_vec = normalise(mem_vec + alpha * person_vec)
Retrieval uses: ORDER BY situated_vec <=> query_vec LIMIT n
"""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS situated_embeddings (
                id        TEXT PRIMARY KEY,
                obs_id    TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
                person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                vector    vector(384) NOT NULL,
                UNIQUE(obs_id, person_id)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_se_person
            ON situated_embeddings(person_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_se_hnsw
            ON situated_embeddings
            USING hnsw (vector vector_cosine_ops)
        """)
