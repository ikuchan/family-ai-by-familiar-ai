"""消えた行を指したままの `superseded_by` を外す。

**`superseded_by` には外部キーが無い。** `parent_id` は
`FOREIGN KEY (parent_id) REFERENCES observations(id) ON DELETE SET NULL` で自動的に外れるが、
`superseded_by` は行が消えても指したまま残る。051 が `self_model` 1068 行を退避表へ移したので、
それを指していた参照が宙に浮いた。

宙に浮いた `superseded_by` は「畳まれている」と読まれるので、想起の母集合から外れたまま
戻らない（`WHERE superseded_by IS NULL`）。指す先が無いなら畳まれていないので、外す。

**この中身は推測を含む。** 051・054 と違って退避表を残していないため、何を掃除したかは
残った状態からの逆算である（2026-09-03・`復旧記録` v0.21）。いま本番に「畳先なし」が 2 件
残っているが、どちらも 2026-07-26 の会話で、翌日 054 が 337 行を退避したときに浮いたものと
読める（053 はその前なので拾えていない）。

この id は 2026-08-21 のダンプに残っていたものと同じである。本番の `schema_migrations` には
既に記録されているので、ここでは流れない。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE observations o SET superseded_by = NULL "
            "WHERE o.superseded_by IS NOT NULL "
            "  AND NOT EXISTS (SELECT 1 FROM observations n WHERE n.id = o.superseded_by)"
        )
