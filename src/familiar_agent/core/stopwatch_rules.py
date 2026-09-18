"""ストップウォッチの規則（知-u・2026-09-18・`設計方針_ストップウォッチ` v0.1）。純関数。

経過の言い方（「2 日 3 時間」「58 分 12 秒」）と `[ストップウォッチ]` の枠。鳴らない・確認しない・黙らない。
寿命（`STOPWATCH_MAX_SEC`・6 時間）が近いことは枠に書く（「あと N 分で自動で止まる」）。
"""

from __future__ import annotations

from datetime import datetime, timedelta

#: 寿命が近いと枠に添える閾（秒）〔仮〕
WARN_BEFORE_SEC = 30 * 60


def elapsed_text(seconds: float) -> str:
    """経過を人の言い方で：1 分未満は秒、1 時間未満は分＋秒、1 日未満は時間＋分、それ以上は日＋時間。"""
    s = max(0, int(seconds))
    if s < 60:
        return f"{s} 秒"
    if s < 3600:
        return f"{s // 60} 分 {s % 60} 秒"
    if s < 86400:
        return f"{s // 3600} 時間 {(s % 3600) // 60} 分"
    return f"{s // 86400} 日 {(s % 86400) // 3600} 時間"


def elapsed(row: dict, at: datetime) -> timedelta:
    end = row.get("stopped_at") or at
    return max(timedelta(0), end - row["started_at"])


def render_frame(
    active: list[dict],
    recently_stopped: list[dict],
    *,
    now: datetime,
    max_sec: float,
) -> str:
    if not active and not recently_stopped:
        return ""
    lines = ["[ストップウォッチ]"]
    for r in active:
        run = elapsed(r, now).total_seconds()
        line = f"- id={r['id']} {r['label']} 経過 {elapsed_text(run)}（{r['started_at'].astimezone(now.tzinfo):%H:%M} から）"
        left = max_sec - run
        if 0 < left <= WARN_BEFORE_SEC:
            line += f"。あと {max(1, int(left // 60))} 分で自動で止まる"
        lines.append(line)
    for r in recently_stopped:
        ago = max(0, int((now - r["stopped_at"]).total_seconds() // 60))
        how = "寿命で自動で止めた" if r.get("expired") else "止めた"
        lines.append(
            f"- id={r['id']} {r['label']} は {ago} 分前に{how}（{elapsed_text(elapsed(r, r['stopped_at']).total_seconds())}）"
        )
    return "\n".join(lines)
