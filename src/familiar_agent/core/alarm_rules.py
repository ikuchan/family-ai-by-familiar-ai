"""アラームの規則（知-q・2026-09-18・`設計方針_アラーム` v0.1）。純関数。タイマー（`timer_rules`）とは別物。

- `resolve_at`：「7:00」「21時半」（ローカル時刻・過ぎていれば翌日）を絶対時刻に。
- `needs_confirmation`：鳴る時刻が静穏時間なら**登録せず一度確かめる**（アラームはこれだけ。黙る・聞かないは無い）。
- `render_frame`：`[アラーム]` の枠（次に鳴るもの・直前に鳴ったもの）。
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

_AT_RE = re.compile(r"^\s*(\d{1,2})(?:[:：時](\d{1,2})?)?\s*(半)?\s*(?:分)?\s*$")
_ZEN = str.maketrans("０１２３４５６７８９", "0123456789")


def resolve_at(at: str, *, now: datetime) -> datetime:
    """「7:00」「21時半」「７時」を絶対時刻に。過ぎていれば翌日。読めなければ ValueError。"""
    m = _AT_RE.match(str(at or "").translate(_ZEN))
    if not m:
        raise ValueError(f"時刻を読めない：{at}")
    hour = int(m.group(1))
    minute = int(m.group(2) or 0) + (30 if m.group(3) else 0)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"時刻の範囲外：{at}")
    due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if due <= now:
        due += timedelta(days=1)
    return due


def needs_confirmation(at: datetime, *, quiet) -> "str | None":
    """静穏時間に鳴るなら確かめる理由。それ以外は None（アラームは黙らないので沈黙中の確認も無い）。"""
    if quiet is not None and quiet.is_quiet(at):
        return f"{at:%H:%M} は静穏時間（{quiet.start_hour}〜{quiet.end_hour} 時）なので、鳴らしてよいか"
    return None


def render_frame(active: list[dict], recently_fired: list[dict], *, now: datetime) -> str:
    if not active and not recently_fired:
        return ""
    lines = ["[アラーム]"]
    for r in active:
        at = r["at"].astimezone(now.tzinfo)
        day = (
            "今日"
            if at.date() == now.date()
            else "明日"
            if at.date() == (now + timedelta(days=1)).date()
            else f"{at:%m/%d}"
        )
        lines.append(f"- id={r['id']} {r['label']} {day} {at:%H:%M} に鳴る")
    for r in recently_fired:
        ago = max(0, int((now - r["fired_at"]).total_seconds() // 60))
        lines.append(f"- id={r['id']} {r['label']} は {ago} 分前に鳴った（もう止まっている）")
    return "\n".join(lines)
