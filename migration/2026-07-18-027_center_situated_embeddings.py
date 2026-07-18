"""Recenter existing situated_embeddings with the global mean mu（平均中心化 C2）。

C2 でコードが situated を `normalise(mem_vec + ALPHA*p_vec - mu)` で作り、recall の
クエリも同じ式で作るようになる。既存の situated 行は中心化前（mu を引いていない）なので、
放置すると**新旧が別空間に混在してコサインが比較不能**になる。ここで既存行を同じ式へ
一括で移す。

mu が未推定（`embedding_means` に global の行が無い）なら**何もしない**。コード側も
mu が無ければ中心化しないので、これで一貫する（空 DB・モデル移行直後で壊れない）。

src を import せず自前完結させる（マイグレーションは過去の一度きりの実行を再現する凍結物）。
ALPHA は person_memory_manager の値の凍結写し。
"""

import numpy as np


ALPHA = 0.30  # 凍結写し（person_memory_manager.ALPHA）


def _normalise(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT dim, vector FROM embedding_means WHERE scope = %s AND scope_key = %s",
            ("global", ""),
        )
        row = cur.fetchone()
        if not row:
            return  # mu 未推定なら中心化しない
        mu_dim = row[0] if not isinstance(row, dict) else row["dim"]
        mu_blob = row[1] if not isinstance(row, dict) else row["vector"]
        mu = np.frombuffer(bytes(mu_blob), dtype=np.float32)

        # 既存 situated を、その観測の生埋め込みと person の視点ベクトルから作り直す。
        cur.execute("""
            SELECT s.id, s.person_id, e.vector AS mem_vec, p.perspective_vec
            FROM situated_embeddings s
            JOIN obs_embeddings e ON e.obs_id = s.obs_id
            LEFT JOIN persons p ON p.id = s.person_id
        """)
        rows = cur.fetchall()

        updates = []
        for r in rows:
            if isinstance(r, dict):
                sid, mem_blob, p_blob = r["id"], r["mem_vec"], r["perspective_vec"]
            else:
                sid, _pid, mem_blob, p_blob = r
            if mem_blob is None:
                continue
            mem_vec = np.frombuffer(bytes(mem_blob), dtype=np.float32)
            if mem_vec.size != mu_dim:
                continue  # 次元が合わない残骸は触らない
            if p_blob is not None:
                p_vec = np.frombuffer(bytes(p_blob), dtype=np.float32)
                if p_vec.size != mu_dim:
                    p_vec = np.zeros(mu_dim, dtype=np.float32)
            else:
                p_vec = np.zeros(mu_dim, dtype=np.float32)
            centered = _normalise(mem_vec + ALPHA * p_vec - mu)
            literal = "[" + ",".join(str(float(x)) for x in centered) + "]"
            updates.append((literal, sid))

        for literal, sid in updates:
            cur.execute(
                "UPDATE situated_embeddings SET vector = %s WHERE id = %s",
                (literal, sid),
            )
