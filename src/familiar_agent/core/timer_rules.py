"""タイマーの規則（知-n・2026-09-15・`設計方針_タイマー` v0.1）。純関数。

- `resolve_due`：「n 分後」か「7:00」（ローカル時刻・過ぎていれば翌日）を絶対時刻に。
- `needs_confirmation`：鳴る時刻が静穏時間に入る／沈黙の依頼が生きているなら、**登録せず一度確かめる**
  （通り抜けて鳴らすのは頼んだ本人が確かめたときだけ）。機械が判定し、LLM は聞くだけ。
- `render_frame`：`[タイマー]` の枠。動いているもの（残り／経過）と直前に鳴ったもの（「止めて」に
  「もう止まっている」と答えられるように）。
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

_AT_RE = re.compile(r"^\s*(\d{1,2})(?:[:：時](\d{1,2})?)?\s*(半)?\s*(?:分)?\s*$")
_ZEN = str.maketrans("０１２３４５６７８９", "0123456789")


def resolve_due(*, after_minutes: "float | None", at: "str | None", now: datetime) -> datetime:
    """どちらか一方を受け、絶対時刻を返す。読めなければ ValueError（理由つき）。"""
    if after_minutes is not None and at:
        raise ValueError("「何分後」と「何時」は片方だけ")
    if after_minutes is not None:
        if float(after_minutes) <= 0:
            raise ValueError("分数は 0 より大きく")
        return now + timedelta(minutes=float(after_minutes))
    if at:
        m = _AT_RE.match(str(at).translate(_ZEN))
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
    raise ValueError("「何分後」か「何時」のどちらかが要る")


def needs_confirmation(due: "datetime | None", *, quiet, silence_active: bool) -> "str | None":
    """確かめてから登録すべき理由。要らなければ None。"""
    if due is None:
        return None  # ストップウォッチは鳴らない
    if silence_active:
        return "いま黙っているよう頼まれているので、鳴らしてよいか"
    if quiet is not None and quiet.is_quiet(due):
        return f"{due:%H:%M} は静穏時間（{quiet.start_hour}〜{quiet.end_hour} 時）なので、鳴らしてよいか"
    return None


def _mmss(delta: timedelta) -> str:
    total = max(0, int(delta.total_seconds()))
    return f"{total // 60}:{total % 60:02d}"


def measure_at(row: dict, at: datetime) -> str:
    """止めた瞬間の値：ストップウォッチは「経過 m:ss」、タイマーは「残り m:ss」。"""
    if row.get("due") is None:
        return f" 経過 {_mmss(at - row['started_at'])}"
    return f" 残り {_mmss(row['due'] - at)}"


def render_frame(
    active: list[dict],
    recently_fired: list[dict],
    *,
    now: datetime,
    recently_stopped: "list[dict] | None" = None,
) -> str:
    if not active and not recently_fired and not recently_stopped:
        return ""
    lines = ["[タイマー]"]
    for r in active:
        due = r.get("due")
        if due is not None:
            lines.append(
                f"- id={r['id']} {r['label']} 残り {_mmss(due - now)}（{due.astimezone(now.tzinfo):%H:%M} に鳴る）"
            )
        else:
            lines.append(f"- id={r['id']} {r['label']} 経過 {_mmss(now - r['started_at'])}")
    for r in recently_fired:
        ago = max(0, int((now - r["fired_at"]).total_seconds() // 60))
        lines.append(f"- id={r['id']} {r['label']} は {ago} 分前に鳴った（もう止まっている）")
    for r in recently_stopped or []:
        ago = max(0, int((now - r["cancelled_at"]).total_seconds() // 60))
        lines.append(
            f"- id={r['id']} {r['label']} は {ago} 分前に止めた（{measure_at(r, r['cancelled_at']).strip()}）"
        )
    return "\n".join(lines)
