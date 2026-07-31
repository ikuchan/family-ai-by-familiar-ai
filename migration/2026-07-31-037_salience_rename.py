"""W を溜めていた表を「顕著性」（salience）へ改名する。

`activation` という語が、実装上5つの別の量に相乗りしていた。この表が持つのは、W へ
上がった記録の**顕著さ**であって、取込値と参照回数から導く「根づき」ではない。取り違え
を断つために、表を `memory_salience`、列を `salience` にする。

この表は撤去予定である（W は O からの派生ビューで毎ターン作り直す設計に変わるため、
溜める形自体が無くなる）。それでも改名するのは、**撤去までのあいだ旧名が残ると、読む側が
「これも根づきの一種か」と取り違える**ためである。

改名するのは名前だけで、**値は変換しない**。索引 `idx_ma_recent` は表に付いたまま移る
（PostgreSQL は表の改名で索引を張り替えない）ので、名前だけ実体に合わせて変えておく。

src を import せず自前完結させる（マイグレーションは過去の一度きりの実行を再現する凍結物）。
"""


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE memory_activation RENAME TO memory_salience")
        cur.execute("ALTER TABLE memory_salience RENAME COLUMN activation TO salience")
        cur.execute("ALTER INDEX IF EXISTS idx_ma_recent RENAME TO idx_ms_recent")
