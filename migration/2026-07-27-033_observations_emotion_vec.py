"""感情軸の一次絞り用に `observations.emotion_vec`（vector(4)）を足す。

[D-想起合成] の多軸 union 一次絞りは各軸にインデックスを要求する。感情の距離
$D^2=\\sum_i \\lambda_i(\\mathrm{logit}(x_{obs,i})-\\mathrm{logit}(x_{mood,i}))^2$ は
**ロジット空間の重み付きユークリッド距離**なので、$\\sqrt{\\lambda_i}$ を畳み込んだ4次元の点を
持てば、pgvector の L2 距離がそのまま $D$ になる（関連軸で既に使っている仕組み）。

PAD 4列（`emotion_p`／`emotion_pn`／`emotion_a`／`emotion_dom`）から導ける値だが、
pgvector の索引を張るには vector 型の列が要るため持つ。$\\lambda_i$ を畳み込んでいるので、
$\\lambda_i$ を変えたときは全行を作り直す。

既存行はこの場で埋める（W1b で PAD が入っている前提）。ε は活性導出・感情距離と共通の
0.001（ロジットの発散を避ける）。

src を import せず自前完結させる（マイグレーションは過去の一度きりの実行を再現する凍結物）。
"""

import math

_EPSILON = 0.001


def _logit(x: float) -> float:
    x = min(max(float(x), _EPSILON), 1.0 - _EPSILON)
    return math.log(x / (1.0 - x))


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE observations ADD COLUMN IF NOT EXISTS emotion_vec vector(4)")
        cur.execute(
            "SELECT id, emotion_p, emotion_pn, emotion_a, emotion_dom "
            "FROM observations WHERE emotion_vec IS NULL"
        )
        rows = cur.fetchall()
        for row in rows:
            oid = row[0] if isinstance(row, tuple) else row["id"]
            pad = ([row[1], row[2], row[3], row[4]] if isinstance(row, tuple)
                   else [row["emotion_p"], row["emotion_pn"], row["emotion_a"], row["emotion_dom"]])
            vec = "[" + ",".join(f"{_logit(v):.6f}" for v in pad) + "]"
            cur.execute("UPDATE observations SET emotion_vec = %s WHERE id = %s", (vec, oid))
        # 一次絞りは L2 距離の近傍検索なので vector_l2_ops。
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_obs_emotion_vec "
            "ON observations USING hnsw (emotion_vec vector_l2_ops)"
        )
