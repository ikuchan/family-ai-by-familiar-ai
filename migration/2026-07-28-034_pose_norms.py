"""定点ごとの「見えの普通」を置くテーブル `pose_norms` を作る。

`知覚在席` §3-4 の見え層。定点ごとに DINOv2 埋め込み（384次元・`dinov2-small`）の EMA を
持ち、現フレームとのコサイン距離を $\\widehat{S}_{景色}$ にする。

**部屋の映像から作る値**なので、`agent_state` の雑多なキーバリューには混ぜず、専用の
テーブルへ置く。

`observations` の 3千件規模と違い、行数は定点の数（実機で3）しかない。**索引は張らない**
（主キーの定点名で1件ずつ引くだけで、近傍検索はしない）。次元は `emotion_vec` と同じく
`vector` 型にする（将来「過去の普通」を探したくなったときに、そのまま索引を足せる）。

`observations` は「何回見たか」で、これが `MIN_OBSERVATIONS`（5回）に届くまで距離を出さない。

src を import せず自前完結させる（マイグレーションは過去の一度きりの実行を再現する凍結物）。
"""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pose_norms (
                pose_name    text PRIMARY KEY,
                embedding    vector(384) NOT NULL,
                observations integer NOT NULL DEFAULT 0,
                updated_at   timestamptz NOT NULL DEFAULT now()
            )
            """
        )
