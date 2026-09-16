"""タイマーを鳴らす（知-n・2026-09-15・`設計方針_タイマー` v0.1）。T が毎 tick 呼ぶ。

`due` を過ぎた未発火・未取消のタイマーを拾い、**先に `fired_at` を打ってから**（二度鳴らさない）
`機器` の求め「タイマー：{label}」を積む。確かめて掛けたもの（`passes_quiet`）は `passes_gate` で
配信ゲート（在席・静穏・沈黙の依頼）を通り抜ける。落ちていた間に due を過ぎたものは「遅れて」を添えて鳴る。
鳴った知らせは保留していた発話も配る（掛けているあいだ黙っていた分・`TIMER_SILENCE`）。
I は時計を見ない——時計を見るのは T だけ。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

LATE_AFTER_SEC = 60.0  # これ以上遅れて鳴るなら「遅れて」を添える


def fire_due(store, dif, *, now: "datetime | None" = None) -> int:
    """鳴らした本数を返す。"""
    now = now or datetime.now(timezone.utc)
    fired = 0
    for r in store.due_now(now=now):
        if not store.mark_fired(int(r["id"]), now=now):
            continue  # 誰かが先に鳴らした・止めた
        late = (now - r["due"]).total_seconds()
        who = str(r.get("asked_by") or "")
        content = f"タイマー：「{r['label']}」の時間" + (f"（{who}に頼まれたもの）" if who else "")
        if late >= LATE_AFTER_SEC:
            content += f"。{int(late // 60)} 分遅れて鳴っている（{r['due'].astimezone():%H:%M} の予定だった）"
        passes = bool(r.get("passes_quiet"))
        # 鳴るまで黙っていたあいだに溜めた返事も、知らせと一緒に配る（`TIMER_SILENCE`）。
        dif.device("タイマー", content, release_pending=True, passes_gate=passes)
        logger.info(
            "タイマーが鳴った id=%s %s 遅れ=%.0f秒 通り抜け=%s", r["id"], r["label"], late, passes
        )
        fired += 1
    return fired
