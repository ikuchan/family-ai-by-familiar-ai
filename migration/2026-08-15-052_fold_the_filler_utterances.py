"""つなぎの発話を、その求めへ畳む。

**「つなぎ」は間をつなぐ一言である**（`つなぎに言った：ちょっと待ってね。` など）。求めの
処理中に「待っててね」と伝えた記録で、単体で想起に出続けるものではない。その求めへ畳めば、
鎖をたどれば残るし、想起の候補からは外れる。

畳む先は `parent_id`＝その求めの鎖の先頭である。つなぎは `parent_id` を持って書かれていた
（`event_loop` の `_say_filler`）。

この id は 2026-08-21 のダンプに残っていたものと同じである。本番の `schema_migrations` には
既に記録されているので、ここでは流れない。**畳んだ行は翌日 054 が退避表へ移したので、
本番のダンプからは 052 の効果を直接は読み取れない。**
"""

from __future__ import annotations

_FILLER = "content LIKE 'つなぎに言った：%'"


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE observations SET superseded_by = parent_id "
            f"WHERE {_FILLER} AND parent_id IS NOT NULL AND superseded_by IS NULL"
        )
