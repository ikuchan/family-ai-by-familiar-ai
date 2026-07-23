"""時刻の生成を一箇所に集める。

どの時計を使うかは保存先の列で決まる。ここはその使い分けを書き出す場所であり、
既存の値を換算する場所ではない。

- **timestamptz 列**（`observations.timestamp` ほか）には `now_utc()`。tz を持たない
  `datetime` を入れると、DB がセッションの TimeZone で解釈し、ローカル時刻の値が
  そのまま UTC として保存される（2026-07-20 に9時間先へずれる不具合が出た）。
- **TEXT 列**（`memory_events.created_at`／`memory_jobs.available_at` ほか）には
  `now_local_iso()`。これらはローカル時刻の ISO 文字列どうしを比較しており
  （`available_at <= now`）、UTC へ換算すると既存行との比較が壊れる。
- **その日の終わり**のように人の生活時間で意味が決まる時刻は `end_of_day_utc()`。

TEXT 列をいずれ timestamptz へ移すなら、その判断はこのモジュールの内側に閉じる。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone, tzinfo


def local_tz() -> tzinfo:
    """このホストのローカルタイムゾーン。生活時間の基準に使う。"""
    return datetime.now().astimezone().tzinfo or timezone.utc


def local_utc_offset() -> str:
    """ローカルの UTC オフセットを `'+09:00'` 形式で返す（DB セッション TZ 設定用）。

    DB セッションの TimeZone をこの値に設定すると、timestamptz→date/時分の変換や
    psycopg2 が返す datetime が生活時間（ローカル）になる。固定オフセットなので、
    DST のある地域では接続存続中にずれうる（JST は DST なしで問題ない）。
    """
    off = datetime.now(local_tz()).utcoffset() or timedelta(0)
    total = int(off.total_seconds())
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    return f"{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}"


def now_utc() -> datetime:
    """timestamptz 列へ入れる現在時刻（tz 付き）。"""
    return datetime.now(timezone.utc)


def now_utc_iso() -> str:
    """TEXT 時刻列へ入れる現在時刻（UTC・tz 付きの ISO 文字列＝`+00:00`）。

    DB は UTC で管理する（保存列はこれを使う）。aware なので `+00:00` が付き、
    naive との取り違えが起きない。
    """
    return datetime.now(timezone.utc).isoformat()


def now_local_iso() -> str:
    """ローカル暦日境界の計算に使う現在時刻（ローカル・tz なしの ISO 文字列）。

    保存列には使わない（保存は `now_utc_iso()`＝UTC）。`timestamp::date >= cutoff`
    のように、セッション TZ でローカル化された timestamptz と**ローカル暦日**で
    突き合わせる計算に限って使う（UTC 化すると日境界が9時間ずれる）。
    """
    return datetime.now().isoformat()


def now_local_str() -> str:
    """プロンプト表示用の現在時刻＝OS のローカル時刻にタイムゾーンを付記した文字列。

    例：`2026-07-23 15:00 JST(+0900)`。DB は UTC で持つが、プロンプト上は OS が持つ
    タイムゾーンを添えて見せる。
    """
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z(%z)")


def end_of_day_utc(date_str: str) -> datetime:
    """`YYYY-MM-DD` の「その日の終わり」をローカル 23:59:59 として UTC で返す。"""
    d = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
    return d.replace(hour=23, minute=59, second=59, tzinfo=local_tz()).astimezone(timezone.utc)


def ts_to_date(ts) -> str:
    """timestamptz の行値を YYYY-MM-DD の文字列にする。"""
    if ts is None:
        return ""
    if isinstance(ts, str):
        return ts[:10]
    return ts.date().isoformat()


def ts_to_time(ts) -> str:
    """timestamptz の行値を HH:MM の文字列にする。"""
    if ts is None:
        return ""
    if isinstance(ts, str):
        return ts[11:16] if len(ts) >= 16 else ts
    return ts.strftime("%H:%M")
