"""毎ターンの `self_model` を退避表へ移す。

**畳まれずに溜まり続けていた。** 2026-08-21 のダンプでは 1068 行（2026-06-08 〜 08-10）が
あり、`superseded_by` が付いていたのは **1068 中 1 件だけ**である。中身は英語の短い自己記述で、
`Nothing.` が 6 件あった——書き込み側の `insight.lower() != "nothing"` という判定を、
ピリオド付きの `Nothing.` がすり抜けていた。

**読み手はいなかった。** `recall_self_model` と `format_self_model_for_context` は本番コード
から呼ばれていない（2026-09-03 に確認）。毎ターン軽量LLM を呼んで書いていたが、書いたものを
誰も読んでいなかった。自己理解は capability manifest と REST 内省（記-a）が担う。

**削除でなく退避する。** 表は `observations` と同じ列を持つ素の写し（制約も索引も無い）で、
戻せる形にしておく。`observations` から消えると、`obs_embeddings` と `situated_memories` の
対応する行は外部キーの CASCADE で一緒に落ちる。

この id は 2026-08-21 のダンプに残っていたものと同じである。本番の `schema_migrations` には
既に記録されているので、ここでは流れない。
"""

from __future__ import annotations

_WHERE = "direction = '内省' AND kind = 'self_model'"


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS observations_removed_self_model "
            "(LIKE observations)"
        )
        cur.execute(
            f"INSERT INTO observations_removed_self_model SELECT * FROM observations WHERE {_WHERE}"
        )
        cur.execute(f"DELETE FROM observations WHERE {_WHERE}")
