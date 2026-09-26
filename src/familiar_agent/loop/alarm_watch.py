"""アラームを鳴らす（知-q・2026-09-18・`設計方針_アラーム` v0.1）。T が毎 tick 呼ぶ。タイマー（`timer_watch`）とは別物。

`at` を過ぎた未発火・未取消を拾い、**先に `fired_at` を打ってから**（二度鳴らさない）音（`DIF.ring`・
`ALARM_RING_SEC`）を鳴らし、`機器` の求め「アラーム：{label}」を積む。静穏時間の音は確かめて掛けたもの
（`passes_quiet`）だけ（声も同じ・`passes_gate`）。落ちていた間に過ぎたものは「遅れて」を添えて鳴る。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

LATE_AFTER_SEC = 60.0


def fire_due(
    store,
    dif,
    *,
    now: "datetime | None" = None,
    ring_sec: float = 0.0,
    quiet: bool = False,
    gain: float = 1.0,
) -> int:
    """鳴らした本数を返す。"""
    now = now or datetime.now(timezone.utc)
    fired = 0
    for r in store.due_now(now=now):
        if not store.mark_fired(int(r["id"]), now=now):
            continue
        late = (now - r["at"]).total_seconds()
        who = str(r.get("asked_by") or "")
        content = f"アラーム：「{r['label']}」の時刻" + (f"（{who}に頼まれたもの）" if who else "")
        if late >= LATE_AFTER_SEC:
            content += f"。{int(late // 60)} 分遅れて鳴っている（{r['at'].astimezone():%H:%M} の予定だった）"
        passes = bool(r.get("passes_quiet"))
        if ring_sec > 0 and (passes or not quiet):
            dif.ring(seconds=ring_sec, gain=gain)
        dif.device("アラーム", content, passes_gate=passes)
        logger.info(
            "アラームが鳴った id=%s %s 遅れ=%.0f秒 通り抜け=%s", r["id"], r["label"], late, passes
        )
        fired += 1
    return fired
