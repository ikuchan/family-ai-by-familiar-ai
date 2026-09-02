"""`observations` から視点列3つを落とす（段5）。

`writer_id`（誰が書いたか）・`subject_id`（誰についてか）・`participants_json`（誰が居たか）
を撤去する。人と記憶の結びつきは situated だけが担う（[D-在席相関/V2]）。

**面を立てる材料そのものは消さない。** 誰がしたこと・誰が居たかは書き込みの瞬間には要る
情報で、落とすのは「観測の行に残しておくこと」だけである。材料は `refresh_situated_memories`
へ引数で渡し、立った面が以後の正になる。

**読み手は段4 までに居なくなっていた。**

| 列 | 最後の読み手 | いつ消えたか |
|---|---|---|
| `writer_id` | 重複判定の30秒窓／`actor` の面の材料 | 段5 で `actor` の面へ移した |
| `participants_json` | `present` の面の材料 | 段5 で引数へ移した |
| `subject_id` | **もともと 0**（SQL で読む箇所なし） | — |

**`subject_id` は写さずに落とせる。** 2026-08-21 のダンプで実在の人を指すのは 397 件だが、
**その全件がその人の面を既に持っている**（`present` 337／`about` 79／`addressee` 35／
`actor` 26／`source` 9／`beneficiary` 2）。写す先が無い。

**この3列は原本の 054 までに落ちていない。** 8月21日のダンプにも残っている。失われた16本の
復元ではなく、situated V2 を閉じるための新規の1本である（`課題8` の段5）。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        for column in ("writer_id", "subject_id", "participants_json"):
            cur.execute(f"ALTER TABLE observations DROP COLUMN IF EXISTS {column}")
