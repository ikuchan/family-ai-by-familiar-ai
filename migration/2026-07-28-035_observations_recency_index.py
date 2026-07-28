"""時間軸の並べ替えの式に索引を張る。

`by_time`（時間軸の一次絞り）は、基準時刻からの隔たりで絞って並べる。その鍵は
`COALESCE(last_recalled_at, timestamp)` で、採点の起点と同じ式である。列ごとの索引
（`idx_obs_timestamp`）はこの式には効かないため、絞りも並べ替えも全走査で答えていた。

**いま速くなるとは限らない。** 対象は数千件で、この規模ではプランナが順次走査を選ぶ
ことが多い（実測でも 1〜2ms で返っている）。記録が増えたときに効く備えである。

`CONCURRENTLY` は使わない。既存のマイグレーションと同じくトランザクションの中で張る
形に揃える（この規模なら一瞬で終わり、書き込みを止める時間も短い）。

src を import せず自前完結させる（マイグレーションは過去の一度きりの実行を再現する凍結物）。
"""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_observations_recency "
            "ON observations (COALESCE(last_recalled_at, timestamp))"
        )
