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

from datetime import datetime, timezone, tzinfo


def local_tz() -> tzinfo:
    """このホストのローカルタイムゾーン。生活時間の基準に使う。"""
    return datetime.now().astimezone().tzinfo or timezone.utc


def now_utc() -> datetime:
    """timestamptz 列へ入れる現在時刻（tz 付き）。"""
    return datetime.now(timezone.utc)


def now_local_iso() -> str:
    """TEXT 列へ入れる現在時刻（ローカル・tz なしの ISO 文字列）。

    既存行と文字列比較するため、tz を付けない。
    """
    return datetime.now().isoformat()


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
