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
行ごとの判定は持たない。二重適用の防止はランナーが `schema_migrations` で
担うので、ここでデータを見て「適用済みか」を推測しない（推測すると、状況次第で
正しい行まで巻き添えでずらしうる）。
"""

# JST は UTC+9 固定（夏時間なし）。移行専用の凍結値。
_SHIFT_HOURS = 9


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE observations "
            "SET timestamp = timestamp - make_interval(hours => %s)",
            (_SHIFT_HOURS,),
        )
