"""内なる記録の `actor` を、エージェント自身へ寄せる。

047 が `writer_id` から素直に `actor` を立てると、話者が解決できなかった記録
（`writer_id` が `default`）の `actor` が `default` になる。だが `default` は
「まだ誰か分からない」の置き場であって、人ではない。

**話者が解決できなかった記録は、パジュ自身がしたことである。** 会話の要約も、観察も、
好奇心も、求めも、記憶も、完了も、意図も、すべてパジュの体験である
（[D-在席相関]「その日の O はすべて自己の体験」）。

2026-08-21 の実物では、`writer_id` が `default` の 2544 件がすべて `actor = __self__`
になっており、situated に `default` の面は1件も無かった（会話 1903・観察 257・
好奇心 238・求め 59・記憶 52・完了 15・意図 5 ほか）。

コード側（`refresh_situated_memories`）は最初からこの規則で立てるので、ここが直すのは
047 が残した既存行だけである。

この id は 2026-08-21 のダンプに残っていたものと同じである。本番の
`schema_migrations` には既に記録されているので、ここでは流れない。
"""

from __future__ import annotations

_AGENT_SELF = "00000000-0000-0000-0000-000000000000"
_DEFAULT = "00000000-0000-0000-0000-000000000001"


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        # 既に自分の面がある観測は、`default` の面を落とすだけでよい。
        cur.execute(
            "DELETE FROM situated_memories sm "
            " WHERE sm.relation_key = 'actor' AND sm.person_id = %s "
            "   AND EXISTS (SELECT 1 FROM situated_memories t "
            "                WHERE t.obs_id = sm.obs_id AND t.relation_key = 'actor' "
            "                  AND t.person_id = %s)",
            (_DEFAULT, _AGENT_SELF),
        )
        cur.execute(
            "UPDATE situated_memories SET person_id = %s "
            " WHERE relation_key = 'actor' AND person_id = %s",
            (_AGENT_SELF, _DEFAULT),
        )
