"""タイマーの規則（知-n・2026-09-15・`設計方針_タイマー` v0.1）。純関数。

- `resolve_due`：「n 分後」を絶対時刻に（「何時に」はアラーム・`alarm_rules`・2026-09-18）。
- `needs_confirmation`：鳴る時刻が静穏時間に入る／沈黙の依頼が生きているなら、**登録せず一度確かめる**
  （通り抜けて鳴らすのは頼んだ本人が確かめたときだけ）。機械が判定し、LLM は聞くだけ。
- `render_frame`：`[タイマー]` の枠。動いているもの（残り／経過）と直前に鳴ったもの（「止めて」に
  「もう止まっている」と答えられるように）。
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

_ZEN = str.maketrans("０１２３４５６７８９", "0123456789")


def resolve_due(*, after_minutes: "float | None", now: datetime) -> datetime:
    """「n 分後」を絶対時刻に。読めなければ ValueError（理由つき）。

    「何時に」はアラーム（`alarm_rules.resolve_at`）。タイマーとアラームは別物（2026-09-18・知-q）。
    """
    if after_minutes is None:
        raise ValueError("「何分後」が要る（何時に、ならアラーム）")
    if float(after_minutes) <= 0:
        raise ValueError("分数は 0 より大きく")
    return now + timedelta(minutes=float(after_minutes))


def needs_confirmation(due: datetime, *, quiet, silence_active: bool) -> "str | None":
    """確かめてから登録すべき理由。要らなければ None。"""
    if silence_active:
        return "いま黙っているよう頼まれている"
    if quiet is not None and quiet.is_quiet(due):
        return f"{due:%H:%M} は静穏時間（{quiet.start_hour}〜{quiet.end_hour} 時）"
    return None


def confirm_text(minutes: float, *, silence: bool, mic_close: bool) -> str:
    """掛ける前の確認の言葉（`TIMER_CONFIRM`・設定に合わせて変わる）。LLM は相手に合わせて言い直してよい。"""
    m = int(minutes) if float(minutes).is_integer() else minutes
    behaviour = {
        (True, True): "その間は黙って聞かないよ",
        (True, False): "その間は黙っているよ",
        (False, True): "その間は聞かないよ",
        (False, False): "",
    }[(bool(silence), bool(mic_close))]
    tail = f"。{behaviour}、いい？" if behaviour else "、いい？"
    return f"{m} 分のタイマーね{tail}"


_CONTROL = re.compile(r"止め|ストップ|中止|やめ|一時停止|再開(?!発)")  # 「再開発」は会話


def is_control_word(text: str) -> bool:
    """聞かないあいだ（`TIMER_MIC_CLOSE`）でも通す、タイマーの操作の言葉か。

    止める・一時停止・再開の語を含むもの。長さでは切らない（ウの 3・2026-11-14・本人の決定）。
    **通すかを決めるだけ**で、何をするかは調停か主LLM が道具で決める（言葉はあいまいでありうる）。
    """
    return bool(_CONTROL.search((text or "").strip()))


def _mmss(delta: timedelta) -> str:
    total = max(0, int(delta.total_seconds()))
    return f"{total // 60}:{total % 60:02d}"


def remaining(row: dict, now: datetime) -> timedelta:
    """タイマーの残り。止めている間は動かない（`paused_at` 基準）。

    `due` は再開のたびに止めていた長さぶん伸びている（`TimerStore.resume`）ので、動いている間は
    `due − now` でよい。止めている間は `due − paused_at`（止めた瞬間の残りのまま）。
    """
    paused_at = row.get("paused_at")
    at = paused_at if paused_at is not None else now
    return max(timedelta(0), row["due"] - at)


def measure_at(row: dict, at: datetime) -> str:
    """止めた瞬間の値：「残り m:ss」（ストップウォッチは別物・`stopwatch_rules`）。"""
    return f" 残り {_mmss(remaining(row, at))}"


def render_frame(
    active: list[dict],
    recently_fired: list[dict],
    *,
    now: datetime,
    recently_stopped: "list[dict] | None" = None,
    not_listening: str = "",
) -> str:
    if not active and not recently_fired and not recently_stopped:
        return ""
    lines = ["[タイマー]"]
    if not_listening:
        # 名前が要ること（出-au 段 1-2）と、時間を聞けること（ウの 3・2026-11-13）も書く。主LLM・軽量LLM と GUI のパネルが読む。
        lines.append(
            f"- 聞いていない（{not_listening} が鳴るまで。名前を呼んだときの、止めて・一時停止・再開の言葉と、"
            "タイマーへの問いだけ届く）"
        )
    for r in active:
        due = r["due"]
        if r.get("paused_at") is not None:
            lines.append(
                f"- id={r['id']} {r['label']} 一時停止中（残り {_mmss(remaining(r, now))}）"
            )
        else:
            lines.append(
                f"- id={r['id']} {r['label']} 残り {_mmss(remaining(r, now))}（{due.astimezone(now.tzinfo):%H:%M} に鳴る）"
            )
    for r in recently_fired:
        ago = max(0, int((now - r["fired_at"]).total_seconds() // 60))
        lines.append(f"- id={r['id']} {r['label']} は {ago} 分前に鳴った（もう止まっている）")
    for r in recently_stopped or []:
        ago = max(0, int((now - r["cancelled_at"]).total_seconds() // 60))
        lines.append(
            f"- id={r['id']} {r['label']} は {ago} 分前に止めた（{measure_at(r, r['cancelled_at']).strip()}）"
        )
    return "\n".join(lines)
