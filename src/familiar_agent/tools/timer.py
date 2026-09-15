"""タイマーの道具（知-n・2026-09-15・`設計方針_タイマー` v0.1）。

主LLM が呼ぶ 3 本——`set_timer`（アラーム・タイマー）・`start_stopwatch`（ストップウォッチ）・
`cancel_timer`（止める・**「途中で停められる」の主な口**）。状態は表 `timers`（`store/timers.py`）、
記憶には `予定` の記録（登録）と「やめた」の記録（取消）を書く。

**静穏時間に掛かる／黙っているよう頼まれているときは、いきなり登録しない。** `confirmed` 無しの
呼び出しには「確かめて」を返し、主LLM が本人に一度聞く。「いい」と言われたら `confirmed=true` で
登録し、その印（`passes_quiet`）で鳴るときに配信ゲートを通り抜ける。機械が判定し、LLM は聞くだけ。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

from ..core import timer_rules
from ..io.oif import MI

logger = logging.getLogger(__name__)

MAX_ACTIVE = 5  # 同時に動かせる本数〔仮〕
RECENT_SEC = 180.0  # 鳴った後も枠に残す秒数〔仮〕（「止めて」に「もう止まっている」と答える）

_DEFS: list[dict] = [
    {
        "name": "set_timer",
        "description": (
            'アラームやタイマーを掛ける。「3分測って」は after_minutes=3、「7時に起こして」は at="7:00"。'
            "label には何のためかを短く。返りに id と鳴る時刻。"
            "返りが「確かめて」なら、まだ掛かっていない——その理由を相手に伝えて一度だけ聞き、"
            "「いい」と言われたら同じ引数に confirmed=true を付けてもう一度呼ぶ。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "after_minutes": {"type": "number", "description": "今から何分後（0 より大きい）"},
                "at": {
                    "type": "string",
                    "description": '何時に（例 "7:00"・"21時半"）。過ぎていれば翌日',
                },
                "label": {"type": "string", "description": "何のため（例「パスタ」「起こす」）"},
                "confirmed": {
                    "type": "boolean",
                    "description": "相手に確かめて「いい」と言われたら true",
                },
            },
            "required": ["label"],
        },
    },
    {
        "name": "start_stopwatch",
        "description": "ストップウォッチを始める（「今から測って」）。鳴らない。経過は [タイマー] の枠で分かる。",
        "input_schema": {
            "type": "object",
            "properties": {"label": {"type": "string", "description": "何を測るか"}},
            "required": ["label"],
        },
    },
    {
        "name": "cancel_timer",
        "description": (
            "動いているタイマーやストップウォッチを止める（「タイマー止めて」「やっぱりいい」）。"
            'id は [タイマー] の枠にある番号。全部止めるなら "all"。'
        ),
        "input_schema": {
            "type": "object",
            "properties": {"id": {"description": '番号か "all"'}},
            "required": ["id"],
        },
    },
]


class TimerTool:
    def __init__(
        self,
        *,
        store: Callable[[], Any],
        oif: Any,
        speaker: Callable[[], str],
        quiet: Callable[[], Any],
        silence_active: Callable[[], bool],
        now: "Callable[[], datetime] | None" = None,
    ) -> None:
        self._store = store
        self._oif = oif
        self._speaker = speaker
        self._quiet = quiet
        self._silence_active = silence_active
        self._now = now or (lambda: datetime.now(timezone.utc).astimezone())

    def store(self):
        """器（`TimerStore`）。T が鳴らすときに使う。"""
        return self._store()

    def get_tool_definitions(self) -> list[dict]:
        return [dict(d) for d in _DEFS]

    def frame(self) -> str:
        """`[タイマー]` の枠。無ければ空。"""
        now = self._now()
        store = self._store()
        return timer_rules.render_frame(
            store.active(now=now), store.recently_fired(now=now, within_sec=RECENT_SEC), now=now
        )

    async def call(self, name: str, tool_input: dict) -> tuple[str, bool]:
        """(文面, 使えたか)。"""
        try:
            if name == "set_timer":
                return await self._set(tool_input)
            if name == "start_stopwatch":
                return await self._start_stopwatch(tool_input)
            if name == "cancel_timer":
                return await self._cancel(tool_input)
        except Exception as e:  # noqa: BLE001
            logger.exception("タイマーの道具に失敗: %s", e)
            return f"タイマーの道具が使えなかった：{e}", False
        return f"そんな道具は無い：{name}", False

    async def _set(self, inp: dict) -> tuple[str, bool]:
        label = str(inp.get("label") or "タイマー").strip()
        now = self._now()
        try:
            due = timer_rules.resolve_due(
                after_minutes=(
                    float(inp["after_minutes"]) if inp.get("after_minutes") is not None else None
                ),
                at=(str(inp["at"]) if inp.get("at") else None),
                now=now,
            )
        except (ValueError, TypeError) as e:
            return f"時刻を読めない：{e}", False
        store = self._store()
        if len(store.active(now=now)) >= MAX_ACTIVE:
            return f"同時に動かせるのは {MAX_ACTIVE} 本まで。先にどれかを止めて", False
        reason = timer_rules.needs_confirmation(
            due, quiet=self._quiet(), silence_active=self._silence_active()
        )
        confirmed = bool(inp.get("confirmed"))
        if reason and not confirmed:
            return (
                f"まだ掛けていない。確かめてから：{reason}——{due:%H:%M} に「{label}」で鳴らしてよいか本人に一度聞き、"
                "「いい」なら confirmed=true で呼び直す",
                True,
            )
        who = self._speaker() or ""
        obs_id = await self._write(
            f"{who or '誰か'}に頼まれて、{due:%H:%M} に「{label}」のタイマーを掛けた"
        )
        tid = store.add(
            label=label, due=due, asked_by=who, obs_id=obs_id, passes_quiet=bool(reason), now=now
        )
        logger.info(
            "タイマーを掛けた id=%d %s due=%s 通り抜け=%s",
            tid,
            label,
            due.isoformat(),
            bool(reason),
        )
        return f"掛けた：id={tid} 「{label}」 {due:%H:%M} に鳴る" + (
            "（静かな時間でも鳴らす）" if reason else ""
        ), True

    async def _start_stopwatch(self, inp: dict) -> tuple[str, bool]:
        label = str(inp.get("label") or "ストップウォッチ").strip()
        now = self._now()
        store = self._store()
        if len(store.active(now=now)) >= MAX_ACTIVE:
            return f"同時に動かせるのは {MAX_ACTIVE} 本まで。先にどれかを止めて", False
        who = self._speaker() or ""
        obs_id = await self._write(
            f"{who or '誰か'}に頼まれて、{now:%H:%M} から「{label}」を測り始めた"
        )
        tid = store.add(
            label=label, due=None, asked_by=who, obs_id=obs_id, passes_quiet=False, now=now
        )
        logger.info("ストップウォッチを始めた id=%d %s", tid, label)
        return f"測り始めた：id={tid} 「{label}」 {now:%H:%M} から", True

    async def _cancel(self, inp: dict) -> tuple[str, bool]:
        now = self._now()
        store = self._store()
        target = inp.get("id")
        if str(target).strip().lower() == "all":
            rows = store.active(now=now)
            if not rows:
                return "動いているタイマーは無い", True
            n = store.cancel_all(now=now)
            await self._write("やめた：" + "・".join(f"「{r['label']}」" for r in rows))
            logger.info("タイマーを全部止めた（%d 本）", n)
            return f"{n} 本止めた：" + "・".join(f"「{r['label']}」" for r in rows), True
        try:
            tid = int(str(target))
        except (TypeError, ValueError):
            return f"id を読めない：{target}", False
        row = next((r for r in store.active(now=now) if int(r["id"]) == tid), None)
        if row is None or not store.cancel(tid, now=now):
            return f"id={tid} のタイマーは動いていない", False
        await self._write(f"やめた：「{row['label']}」のタイマー")
        logger.info("タイマーを止めた id=%d %s", tid, row["label"])
        return f"止めた：id={tid} 「{row['label']}」", True

    async def _write(self, content: str) -> "str | None":
        try:
            return await self._oif.write(
                MI(id="", content=content[:500], timestamp=None, direction="予定")
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("予定の記録を書けなかった: %s", e)
            return None
