"""つなぎの発話を記憶から外し、退避表へ移す。

**記録には理由が書かれていたが、その理由は別の仕組みが満たしていた。** 書き込み側には
「残さないと、次の反復の W に『もう一言伝えた』事実が入らず、調停はそれを知らないまま同じ
ことをまた言う（実機で1秒差に同じ文が2回出た）」とあった。だが `_said_fillers` が
**プロンプトへ直接載る**ので（「すでに相手へ伝えた一言」）、O に残さなくても次の反復には
伝わる。**二重に持っていた**ぶんを外す。

2026-08-21 のダンプでは 337 行がここに入っており、すべて `direction='発話' AND
kind='observation'` で本文が `つなぎに言った：` で始まる。

**削除でなく退避である。** 表は `observations` と同じ列を持つ素の写し（制約も索引も無い）。
`observations` から消えると、`obs_embeddings` と `situated_memories` の対応する行は外部キーの
CASCADE で一緒に落ちる。

この id は 2026-08-21 のダンプに残っていたものと同じである。本番の `schema_migrations` には
既に記録されているので、ここでは流れない。
"""

from __future__ import annotations

_FILLER = "content LIKE 'つなぎに言った：%'"


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS observations_removed_fillers (LIKE observations)"
        )
        cur.execute(
            f"INSERT INTO observations_removed_fillers SELECT * FROM observations WHERE {_FILLER}"
        )
        cur.execute(f"DELETE FROM observations WHERE {_FILLER}")
