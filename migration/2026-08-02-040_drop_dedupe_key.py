"""`memory_events.dedupe_key` を落とす。

重複防止は2つあった。鍵による突き合わせ（この列・キューへ積む段）と、時間窓
（`observations` への書き込み時・30秒・内容と kind の一致）である。時間窓へ一本化する。

鍵が実際に弾いていたのは、同じターン内で同じ内容を二度書く場面（会話 summary・好奇心・
自己モデル）で、いずれも30秒に収まる。日次要約については元から効いていなかった。鍵の
digest は content の sha1 だが、要約は生成のたびに中身が違うので一致しない。実データでも
1日に複数の日次要約が残っており、8月20日の51件は中身が51通りだった。

この列は 記-d の一覧に無いが、039・041・042 と同じ 2026-08-03 14時54分（日本時間）に
流れている。なぜ落としたかの記録は残っていない。

この id は 2026-08-21 のダンプに残っていたものと同じである。本番の
`schema_migrations` には既に記録されているので、ここでは流れない。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        # 索引は DROP COLUMN が連れて落ちるが、名前で残っていた場合に備えて先に落とす。
        cur.execute("DROP INDEX IF EXISTS idx_memory_events_dedupe")
        cur.execute("ALTER TABLE memory_events DROP COLUMN IF EXISTS dedupe_key")
