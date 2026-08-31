"""`unfinished_business` 表を落とす（記-d）。

開いた意図は O の記録（`direction="求め"`）が担っており、この表は二重の実装だった。
用語一覧は開いた意図を「O の MI（status=open）で表し想起で W に上がる」と定めており、
いまのループはそのとおり動いている。

書き手は `heartbeat._persist_remainder` の1箇所だけで、それを呼ぶ `apply_status` は
環-c（旧 `run()` の撤去・2026-07-29）で呼び出し側を失っていた。読み手は本番コードに
0件（`list_unfinished_business` を呼ぶのはテスト1件のみ）だった。

索引（`idx_ub_status`・`idx_unfinished_business_person`）と外部キー2本は
`DROP TABLE` が連れて落ちるので、個別に落とさない。

この id は 2026-08-21 のダンプに残っていたものと同じである。本番の
`schema_migrations` には既に記録されているので、ここでは流れない。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS unfinished_business")
