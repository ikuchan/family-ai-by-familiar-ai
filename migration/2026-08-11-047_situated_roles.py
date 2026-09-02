"""関係エッジを「全登録人物への無差別コピー」から「関係のある人だけ」へ変える。

**面の生成は二段である。**

  段① 機械（047・048）  `actor`（誰がやったか）←`writer_id`／
                         `present`（誰が居たか）←`participants_json`
  段② REST 内省（記-a）  `addressee`／`about`／`experiencer`／`beneficiary`／
                         `companion`／`source`／`owner` …
                         本文を読んで意味役割を抽出し、**既存の観測にもさかのぼって**
                         足していく（[D-在席相関/V2]「語彙は REST が育て・畳む」）。

このマイグレーションは段①だけを立てる。段②は REST の仕事で、ここでは立てない。

**それまでの生成は無差別だった。** 観測1件につき登録人物全員＋AGENT_SELF の行を
`presence` 固定で作っており（6433×6≈38,600 行）、全員に同じベクトルが入るので
「その人がその記憶とどう関わったか」を表していなかった。関係のある人だけにすると
約 6,806 行（約5.7分の1）になる。

**面の `content` は `[役割の札] ` ＋ 出来事の本文**。`actor` だけ持たない（全観測に
立つので書き直す意味がない）。

既定の関係名を `'presence'` から `'present'` へ改める（2026-08-21 の実物に合わせる）。

この id は 2026-08-21 のダンプに残っていたものと同じである。本番の
`schema_migrations` には既に記録されているので、ここでは流れない。
"""

from __future__ import annotations

def upgrade(conn) -> None:
    with conn.cursor() as cur:
        # 既定の関係名を改める（列の既定値と既存行の両方）。
        cur.execute(
            "ALTER TABLE situated_memories ALTER COLUMN relation_key SET DEFAULT 'present'"
        )
        cur.execute(
            "UPDATE situated_memories SET relation_key = 'present' "
            "WHERE relation_key = 'presence'"
        )

        # `actor` を視点列から立てる（話者未解決なら自分＝048 の規則を先取りしない。
        # ここでは素直に `writer_id` を使い、048 が寄せる）。
        cur.execute(
            "INSERT INTO situated_memories (id, obs_id, person_id, vector, relation_key) "
            "SELECT gen_random_uuid()::text, o.id::text, o.writer_id::text, s.vector, 'actor' "
            "  FROM observations o "
            "  JOIN LATERAL (SELECT vector FROM situated_memories "
            "                 WHERE obs_id = o.id::text LIMIT 1) s ON TRUE "
            " WHERE o.writer_id IS NOT NULL "
            "ON CONFLICT (obs_id, person_id, relation_key) DO NOTHING"
        )

        # `present` に札つきの言葉を入れる（在席者ぶんだけ残す）。
        cur.execute(
            "UPDATE situated_memories sm "
            "   SET content = '[そばに居た] ' || o.content "
            "  FROM observations o "
            " WHERE o.id::text = sm.obs_id AND sm.relation_key = 'present' "
            "   AND o.participants_json::jsonb ? sm.person_id"
        )

        # 関係の無い `present` を落とす（在席者でない人の行）。
        cur.execute(
            "DELETE FROM situated_memories sm "
            " USING observations o "
            " WHERE o.id::text = sm.obs_id AND sm.relation_key = 'present' "
            "   AND NOT (o.participants_json::jsonb ? sm.person_id)"
        )
