"""活性の列を「根づき」（groundedness）へ改名する。

`activation` という語が、実装上5つの別の量に相乗りしていた。取込値と参照回数から導き
**時間では減らない**この量を「根づき」と呼び、記号を `g`、英語を `groundedness` にする。
英語を `entrenchment` にしないのは、頭文字 `e` が想起スコアの `e` 軸（感情一致）と衝突し、
文中で取り違えるためである。

改名するのは名前だけで、**値は変換しない**。`activation_a0` は取込時の初期値、
`activation_n` は参照回数の正味デルタで、どちらも意味は変わらない。

`RENAME COLUMN` はカタログの書き換えだけで、行の書き換えを伴わない。索引と既定値は
列に付いたまま移る。

src を import せず自前完結させる（マイグレーションは過去の一度きりの実行を再現する凍結物）。
"""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE observations RENAME COLUMN activation_a0 TO groundedness_g0"
        )
        cur.execute(
            "ALTER TABLE observations RENAME COLUMN activation_n TO groundedness_n"
        )
