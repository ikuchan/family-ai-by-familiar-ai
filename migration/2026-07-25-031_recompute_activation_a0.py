"""既存の取込 a0（内容の新規性）を、正確な近傍で一括再計算する。

`content_novelty` は「AGENT_SELF 視点の situated 近傍 K 件の平均コサインの裏返し」で
novelty を測り、`a0 = clip(w_n * novelty, 0, cap)` として保存する。ところがその近傍検索は
`person_id`・`superseded_by`・`kind <> 'self_model'` で絞り込んだベクトル検索で、HNSW の
絞り込み検索は近傍候補を集めた後に絞り込みを当てるため、本当の近傍が取れていなかった。
近傍が K 件に満たなければ既定値へ落ち、揃っても本当の近傍でなければ平均コサインが低く出て
novelty が過大になる。実機では `パパが2026年ワールドカップ決勝戦を観た` に a0=1.000
（正確な近傍では 0.674）が付き、名前の羅列でできた場面記録が軒並み a0≈0.95 になっていた。
a0 は想起スコアの加算部で最大の重み（w_a=1.5）を持つため、内容の薄い記録が上位を占めた。

ここでは既存行を**全走査の正確な近傍**で計算し直す。近傍探索の母集合は取込時と同じ
（AGENT_SELF 視点・生きている・`self_model` を除く）とし、**その記録より前の記録だけ**を
見て取込時を再現する。ただし当時 superseded でなかった行が今は superseded ということは
ありうるので、再現は近似である。

src を import せず自前完結させる（マイグレーションは過去の一度きりの実行を再現する凍結物）。
定数は `MemoryConfig` の既定値の凍結写し。
"""

import numpy as np

AGENT_SELF_ID = "00000000-0000-0000-0000-000000000000"  # 凍結写し
NOVELTY_K = 7            # 凍結写し（MemoryConfig.novelty_k）
NOVELTY_W_N = 1.5        # 凍結写し（MemoryConfig.novelty_w_n）
NOVELTY_DEFAULT = 0.5    # 凍結写し（MemoryConfig.novelty_default）
NOVELTY_A0_CAP = 1.5     # 凍結写し（MemoryConfig.novelty_a0_cap）
_CHUNK = 256             # 全対全コサインを分割して計算する幅（メモリを抑える）


def _parse(vec) -> "np.ndarray | None":
    """pgvector の `[0.1,0.2,...]` 表現を float32 配列にする。"""
    if vec is None:
        return None
    if isinstance(vec, (bytes, bytearray)):
        vec = bytes(vec).decode()
    text = str(vec).strip()
    if not text.startswith("["):
        return None
    try:
        return np.fromstring(text[1:-1], sep=",", dtype=np.float32)
    except Exception:
        return None


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        # 対象＝AGENT_SELF 視点の situated を持つ生きている観測。時刻順に並べる。
        cur.execute(
            "SELECT o.id, o.kind, s.vector "
            "FROM observations o JOIN situated_embeddings s ON s.obs_id = o.id "
            "WHERE s.person_id = %s AND o.superseded_by IS NULL "
            "ORDER BY o.timestamp",
            (AGENT_SELF_ID,),
        )
        rows = cur.fetchall()

    parsed = []
    for row in rows:
        oid = row["id"] if isinstance(row, dict) else row[0]
        kind = row["kind"] if isinstance(row, dict) else row[1]
        vec = _parse(row["vector"] if isinstance(row, dict) else row[2])
        if vec is None or vec.size == 0:
            continue
        parsed.append((oid, kind, vec))
    if not parsed:
        return  # 空 DB・視点未整備でも壊れない

    dims = [v.size for _, _, v in parsed]
    dim = max(set(dims), key=dims.count)
    parsed = [(o, k, v) for o, k, v in parsed if v.size == dim]
    if not parsed:
        return

    matrix = np.stack([v for _, _, v in parsed])
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9, None)
    # 近傍探索の母集合は self_model を除く（取込時と同じ）。時刻順なので添字が前後関係。
    in_pool = np.array([k != "self_model" for _, k, _ in parsed], dtype=bool)

    updates = []
    for start in range(0, len(parsed), _CHUNK):
        stop = min(start + _CHUNK, len(parsed))
        sims = matrix[start:stop] @ matrix.T          # (chunk, 全件)
        for offset in range(stop - start):
            i = start + offset
            # 自分より前で、母集合に入る記録だけを見る。
            usable = np.flatnonzero(in_pool[:i])
            if usable.size < NOVELTY_K:
                novelty = NOVELTY_DEFAULT             # 近傍が K 未満は既定
            else:
                nearest = np.sort(sims[offset, usable])[-NOVELTY_K:]
                novelty = 1.0 - float(nearest.mean())
            a0 = max(0.0, min(NOVELTY_A0_CAP, NOVELTY_W_N * novelty))
            updates.append((float(a0), parsed[i][0]))

    with conn.cursor() as cur:
        for a0, oid in updates:
            cur.execute("UPDATE observations SET activation_a0 = %s WHERE id = %s", (a0, oid))
