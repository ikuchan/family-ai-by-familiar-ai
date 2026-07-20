"""Shift observation timestamps back by the local UTC offset（時刻の9時間ずれの補正）。

`observations.timestamp` は tz を持たない `datetime.now()` で書かれていた。列は
`timestamptz` なので、JST の壁掛け時計の値がそのまま UTC として解釈され、実時刻
より9時間先に保存されていた。書き込み側は tz-aware に直したので、既存行をここで
同じ時計へ寄せる。

同じ表の `last_recalled_at` は SQL の `now()` で書かれており正しいので触らない。
放置すると、想起の起点が `timestamp`（9時間先）から `last_recalled_at`（正しい
現在）へ移った瞬間に9時間後退し、強化B（想起で新しさが若返る）と逆に動く。

補正量は固定の9時間。サーバの TZ は一貫して JST（UTC+9・夏時間なし）だったことを
確認したうえで実行する。

全行を一律にずらしてよいのは、**この移行の時点で存在する行が、すべて壊れた
書き込み側で作られたもの**だからである（修正コードとこの移行は同時に入る）。

二重適用は二段で防ぐ。

1. ランナー（`db_migrations`）が `schema_migrations` で適用済みを記録し、二度
   実行しない。これが主たる保証である。
2. 本体でも前提条件を確認する。壊れている間は直近の書き込みが必ず未来にあるので、
   `max(timestamp) > now()` が偽なら何もしない。ランナーを介さない手動実行への保険。

2 には限界がある。相対的なずらしは、補正後のデータと最初から正しかったデータを
見分けられない。よって前提条件が真のとき `UPDATE` は全行に当たり、壊れた行と
正しい行が混在していれば正しい行も動く。本番でその混在は起きない（修正コードと
この移行が同時に入るため、移行時点の行はすべて壊れた側である）。
"""

# JST は UTC+9 固定（夏時間なし）。移行専用の凍結値。
_SHIFT_HOURS = 9


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        # 前提条件：壊れている間は直近の書き込みが未来にある。補正後は偽になるので、
        # 手動で二度走らせても2回目は何もしない（限界は docstring のとおり）。
        cur.execute("SELECT max(timestamp) > now() AS broken FROM observations")
        row = cur.fetchone()
        broken = row[0] if not isinstance(row, dict) else row["broken"]
        if not broken:
            return

        cur.execute(
            "UPDATE observations "
            "SET timestamp = timestamp - make_interval(hours => %s)",
            (_SHIFT_HOURS,),
        )
