"""`situated_memories.relation_key` の既定値を `presence` から `present` へ揃える。

**047 の原本は既定値を `presence` のままにしていた。** 2026-08-21 のダンプの
`situated_memories.relation_key` は `DEFAULT 'presence'` である。復元した 047 が `present`
へ変えたため、本番（`presence`）とテスト（`present`）でスキーマが食い違っていた
（2026-09-02 に、テスト DB とダンプの機械差分で見つけた・`復旧記録` v0.18）。

**役割名は 047 以降すべて `present` に揃っている。** 実データに `presence` の行は 1 件も
無く、全行が明示的に値を入れているので既定値を使う経路は無い。**既定値だけが古い名前で
残っていた**ので、そちらを実物へ合わせる。

これは失われた16本の復元ではなく、復元の過程で生じた食い違いを埋める新規の1本である。
原本に 055 は無いので、連番の続きを取った。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE situated_memories ALTER COLUMN relation_key SET DEFAULT 'present'"
        )
