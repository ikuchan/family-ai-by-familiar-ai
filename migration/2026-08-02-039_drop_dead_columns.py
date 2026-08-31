"""`observations` から読み手の居ない2列を落とす（記-d）。

**`importance`** は P-1 で「日次減衰は使わない・時間減衰は t 軸へ一元化」と決めたときに
役目を失った。値は 021 が `groundedness_g0` へ移してあり、MI の組み立てもそちらを読む。
書き手 `decay_importance` は本番からの呼び出しが0件だったので、あわせて撤去した。

**`scope`** は書くだけで誰も読んでいない。SELECT にも `columns=` 指定にも WHERE にも
現れない。道具 `remember` の `scope` 引数は別物で、「誰のぶんを書くか」を分岐させる制御
である。分岐が決めた相手は `writer_id`／`subject_id`／`participants_json` に残るので、
列を落としても情報は失われない。

索引 `idx_obs_scope` は `DROP COLUMN` が連れて落ちるので、個別に落とさない。

この id は 2026-08-21 のダンプに残っていたものと同じである。本番の
`schema_migrations` には既に記録されているので、ここでは流れない。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE observations DROP COLUMN IF EXISTS importance")
        cur.execute("ALTER TABLE observations DROP COLUMN IF EXISTS scope")
