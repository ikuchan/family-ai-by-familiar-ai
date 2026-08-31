"""`observations.person_id`（所有者絞り）を落とす（記-d）。

010 が「人ごとの記憶空間」を意図して入れた列だが、実データでは機能していなかった。
2026-08-03 時点の 5080 行のうち 4904 行（96.5%）が既定値の `default` のままで、
家族4人のうち2人（いくながこうき・いくながたえこ）は所有行を1件も持たない。

人ごとの区別は situated 側（`situated_embeddings.person_id`）が担っている。
そちらには `default` の行が1件も無く、`__self__` と実在の4人だけが並ぶ。
C-1（`_read_observations_by_situated`）が読み出しをその JOIN へ移した。

C-1 はフォールバック二関数に「situated 行を持たない観測を拾う」役目を残したが、
その母集合は 0 行である（生存する観測 2672 件すべてが situated 行を持つ）。
役目が消えたので、フォールバックからも所有者絞りを外す。

重複判定の30秒窓だけは絞りを保ち、所有者ではなく `writer_id` で絞る。重複とは
「同じ書き手が同じ内容を同じ kind で窓の内に」であって、家族の二人が同じ言葉を
言ったものは重複ではない。

索引 `idx_observations_person` と `persons(id)` への外部キーは `DROP COLUMN` が
連れて落ちるので、個別に落とさない。

この id は 2026-08-21 のダンプに残っていたものと同じである。本番の
`schema_migrations` には既に記録されているので、ここでは流れない。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE observations DROP COLUMN IF EXISTS person_id")
