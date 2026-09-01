"""`situated_embeddings` を `situated_memories` へ変える（索引を記憶にする）。

044 より前の situated は `(id, obs_id, person_id, vector, relation_key)` だけを持つ
**ベクトル索引**で、記憶の実体（本文・時間の起点・根づき）は `observations` にあった。
ここで `content`・`last_recalled_at`・`groundedness_n` を面が自分で持つようにする。
名前の変更（`embeddings` → `memories`）がその移り変わりを表す。

**なぜ面へ移すか**（`設計図` [D-在席相関/V2]・`MIデータモデル` §5）。
一つの出来事を1行だけで持つと、supersede で畳んだ瞬間に「誰が何を言ったのか」が
畳んだ側の `content` の文字列にしか残らない。文字列は版が進むたび書き直されるので、
正確には復元できない。求めの版チェーンは「質問される → 調査する → 結果をまとめて出力する
→ 当初の質問と調査結果を畳んで回答を作る」と進むので、この取りこぼしが実際に起きる。

面（`(obs_id, person_id, relation_key)`）を別々の記憶として残せば、畳んでも面は残る
（`superseded_by` は `observations` の条件で、面の行はそのまま生き続ける）。実データでも
`direction="求め"` の畳まれた版 1005 観測に 1050 の面が残っている。

**何が出来事ごとで、何が面ごとか。** 取込の驚き `groundedness_g0` は**パジュにとっての
驚き**で、取り込んだ瞬間に1回だけ測るものなので出来事に残す。`content`・時間の起点・
根づきの `n` は、どの面を通って思い出したかで変わるので面に付ける。主体はパジュ一人で、
`person_id` は所有者でなく**誰と関係する面か**を表す。

**旧値は引き継がない。** 出来事1件の値をどの面へ写すかに正解が無いためである。原本も
そうしており、2026-08-21 のダンプでは `situated_memories.last_recalled_at` の 139 件が
すべて 044 の適用時刻（2026-08-11 12:41 UTC）より後だった（最小 8月13日・最大 8月18日）。
8月3日時点で観測が持っていた 713 件の起点と 88 件の `n` は、ここで捨てられる。

索引名の追随（`idx_se_person`・`idx_se_hnsw` ほか）は 046 が引き取る。ここで作るのは
新しい列に対する `idx_situated_recency` だけである。

この id は 2026-08-21 のダンプに残っていたものと同じである。本番の
`schema_migrations` には既に記録されているので、ここでは流れない。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE situated_embeddings RENAME TO situated_memories")
        cur.execute("ALTER TABLE situated_memories ADD COLUMN IF NOT EXISTS content TEXT")
        cur.execute(
            "ALTER TABLE situated_memories "
            "ADD COLUMN IF NOT EXISTS last_recalled_at TIMESTAMPTZ"
        )
        cur.execute(
            "ALTER TABLE situated_memories "
            "ADD COLUMN IF NOT EXISTS groundedness_n INTEGER NOT NULL DEFAULT 0"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_situated_recency "
            "ON situated_memories (person_id, last_recalled_at)"
        )
        cur.execute("ALTER TABLE observations DROP COLUMN IF EXISTS last_recalled_at")
        cur.execute("ALTER TABLE observations DROP COLUMN IF EXISTS groundedness_n")
