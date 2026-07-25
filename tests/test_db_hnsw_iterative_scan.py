"""絞り込み付きベクトル検索の取りこぼし対策（pgvector 0.8 の反復スキャン）。

HNSW 索引は `situated_embeddings.vector` 単体に張られている。想起は `person_id` と
`superseded_by IS NULL` で絞り込むため、索引が先に集めた近傍候補（`ef_search` 既定 40）の
大半が他人分で落ち、母集合が 2707 件あっても 0〜1 件しか残らないことがある（実機で観測）。
`hnsw.iterative_scan` は、絞り込みを通った行が必要数に達するまで走査を続ける。
"""

from __future__ import annotations

from familiar_agent.db import get_db


def _setting(name: str) -> str:
    database = get_db()
    with database.lock:
        conn = database.conn()
        with conn.cursor() as cur:
            # pgvector の GUC は、そのセッションで vector を一度触るまで見えない。
            cur.execute("SELECT '[1,0]'::vector <=> '[0,1]'::vector")
            cur.fetchone()
            cur.execute(f"SHOW {name}")
            row = cur.fetchone()
    return str(row[name] if isinstance(row, dict) else row[0])


def test_connection_enables_hnsw_iterative_scan() -> None:
    assert _setting("hnsw.iterative_scan") == "relaxed_order"
