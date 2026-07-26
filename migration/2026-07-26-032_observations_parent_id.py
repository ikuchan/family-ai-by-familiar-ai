"""observations に `parent_id` を足す（親子2階層の MI）。

調査は複数並行しうるので、ループの状態は1本の鎖では表せない。**親**（人の求め・情動）と
**子**（その求めのために投げた調査）の2階層を持ち、**孫は作らない**。親を閉じるときに
生きている子を全部閉じる（一段だけ・再帰なし）ので、DAG にならず再帰も要らない。

`superseded_by` と同じ性質の**構造の列**である。[D-MIモデル] が禁じているのは意味
（意図／未応答／由来／動作）の属性化であって、構造リンクは `supersedes` として既に列で
持っている。意味は従来どおり content に置き、LLM が解釈する。

`ON DELETE SET NULL`：親が消えても子は残す（子は子で意味を持つ記録なので、道連れにしない）。

src を import せず自前完結させる（マイグレーションは過去の一度きりの実行を再現する凍結物）。
"""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE observations "
            "ADD COLUMN IF NOT EXISTS parent_id text "
            "REFERENCES observations(id) ON DELETE SET NULL"
        )
        # 親から子を引く経路（親を閉じるときに生きた子を集める）。FK 列には index を張る。
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_observations_parent "
            "ON observations (parent_id) WHERE parent_id IS NOT NULL"
        )
