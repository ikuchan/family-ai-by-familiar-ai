"""端末の時計アプリへの頼み（端-c c6a・課題9 b2・§8.8）。純関数。

端末の求めの `set_timer`・`set_alarm` は、家のタイマーとアラームに入れず、頼んだ端末の時計アプリで掛ける。
ここでは道具の入力を、端末へ送る字（`timer`：秒と名前、`alarm`：時と分と名前）に直す。時刻の読み方と名前の検めは
家の道具と同じもの（`timer_rules`・`alarm_rules`・`label_rules`）を使う。
"""

from __future__ import annotations

from datetime import datetime

from . import alarm_rules, timer_rules
from .label_rules import clean_label

#: 時計アプリが受けるタイマーの長さの上限（Android の `AlarmClock.EXTRA_LENGTH` は 1〜86400 秒）
TIMER_MAX_SEC = 86_400


def clock_order(action: str, tool_input: dict, *, now: datetime) -> "tuple[str, dict] | str":
    """(端末へ送る種類, 中身)。掛けられないときは、その理由の文。"""
    if action == "set_timer":
        label = clean_label(tool_input.get("label"), "タイマー")
        if tool_input.get("at"):
            return "何時に、はアラーム（set_alarm）。タイマーは何分後だけ"
        try:
            raw = tool_input.get("after_minutes")
            due = timer_rules.resolve_due(
                after_minutes=float(raw) if raw is not None else None, now=now
            )
        except (ValueError, TypeError, OverflowError) as e:
            return f"時刻を読めない：{e}"
        seconds = round((due - now).total_seconds())
        if not 1 <= seconds <= TIMER_MAX_SEC:
            return "端末のタイマーは 1 秒から 24 時間まで"
        return "timer", {"seconds": seconds, "label": label}
    if action == "set_alarm":
        label = clean_label(tool_input.get("label"), "アラーム")
        try:
            at = alarm_rules.resolve_at(str(tool_input.get("at") or ""), now=now)
        except ValueError as e:
            return f"時刻を読めない：{e}"
        return "alarm", {"hour": at.hour, "minute": at.minute, "label": label}
    return f"端末の時計では使えない道具：{action}"
