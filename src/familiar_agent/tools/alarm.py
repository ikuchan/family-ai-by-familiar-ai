"""アラームの道具（知-q・2026-09-18・`設計方針_アラーム` v0.1）。タイマー（`tools/timer.py`）とは別物。

主LLM が呼ぶ 2 本——`set_alarm`（何時に）・`cancel_alarm`。調停（軽量LLM）の候補にも同じ 2 本が載る
（どちらに行っても同じ道具）。遠い時刻に起こす・知らせるもので、それまでは普通に暮らす：黙らない・
聞かない状態にしない・掛ける前の確認は**静穏時間に鳴るときだけ**・同時 5 本〔仮〕・一時停止なし。
共有するのは基盤だけ（音の再生 `DIF.ring`・T の tick・`予定` の記録・静穏時間の判定）。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from ..core import alarm_rules
from ..io.oif import MI
from ..person_memory_manager import AGENT_SELF_ID

logger = logging.getLogger(__name__)

MAX_ACTIVE = 5  # 同時に掛けられる本数〔仮〕（朝と昼など）
RECENT_SEC = 180.0  # 鳴った後も枠に残す秒数〔仮〕

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "set_alarm",
        "description": (
            "アラームを掛ける（「7 時に起こして」「21 時に薬って言って」）。at は何時に（ローカル時刻・過ぎていれば翌日）。"
            "「確かめて」が返ったら本人に一度聞き、「いい」と言われたら同じ引数に confirmed=true を付けてもう一度呼ぶ。"
            "何分後に鳴らすならタイマー（set_timer）。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "at": {
                    "type": "string",
                    "description": '何時に（例 "7:00"・"21時半"）。過ぎていれば翌日',
                },
                "label": {"type": "string", "description": "何のため（例「起こす」「薬」）"},
                "confirmed": {
                    "type": "boolean",
                    "description": "相手に確かめて「いい」と言われたら true",
                },
            },
            "required": ["at", "label"],
        },
    },
    {
        "name": "cancel_alarm",
        "description": '掛かっているアラームを止める・取り消す。id は [アラーム] の枠にある番号。全部なら "all"。',
        "input_schema": {
            "type": "object",
            "properties": {"id": {"description": '番号か "all"'}},
            "required": ["id"],
        },
    },
]


class AlarmTool:
    def __init__(
        self,
        *,
        store: Callable[[], Any],
        oif: Any,
        speaker: Callable[[], str],
        quiet: Callable[[], Any],
        now: "Callable[[], datetime] | None" = None,
        on_cancel: "Callable[[], None] | None" = None,
    ) -> None:
        self._store = store
        self._oif = oif
        self._speaker = speaker
        self._quiet = quiet
        self._now = now or (lambda: datetime.now(timezone.utc).astimezone())
        self._on_cancel = (
            on_cancel  # 止める頼みで音を止める（鳴った後は active に無いので先に呼ぶ）
        )

    def store(self):
        return self._store()

    def get_tool_definitions(self) -> list[dict]:
        return [dict(d) for d in TOOL_DEFINITIONS]

    def frame(self) -> str:
        """`[アラーム]` の枠。無ければ空。"""
        now = self._now()
        store = self._store()
        return alarm_rules.render_frame(
            store.active(now=now), store.recently_fired(now=now, within_sec=RECENT_SEC), now=now
        )

    async def call(self, name: str, tool_input: dict) -> tuple[str, bool]:
        try:
            if name == "set_alarm":
                return await self._set(tool_input)
            if name == "cancel_alarm":
                return await self._cancel(tool_input)
        except Exception as e:  # noqa: BLE001
            logger.exception("アラームの道具に失敗: %s", e)
            return f"アラームの道具が使えなかった：{e}", False
        return f"そんな道具は無い：{name}", False

    async def _set(self, inp: dict) -> tuple[str, bool]:
        label = str(inp.get("label") or "アラーム").strip()
        now = self._now().astimezone()
        try:
            at = alarm_rules.resolve_at(str(inp.get("at") or ""), now=now)
        except ValueError as e:
            return f"時刻を読めない：{e}", False
        store = self._store()
        if len(store.active(now=now)) >= MAX_ACTIVE:
            return f"同時に掛けられるのは {MAX_ACTIVE} 本まで。先にどれかを止めて", False
        reason = alarm_rules.needs_confirmation(at, quiet=self._quiet())
        confirmed = bool(inp.get("confirmed"))
        if reason and not confirmed:
            return (
                f"まだ掛けていない。確かめてから：{reason}——{at:%H:%M} に「{label}」で鳴らしてよいか本人に一度聞き、"
                "「いい」なら confirmed=true で呼び直す",
                True,
            )
        who = self._speaker() or ""
        obs_id = await self._write(
            f"{who or '誰か'}に頼まれて、{at:%m/%d %H:%M} に「{label}」のアラームを掛けた"
        )
        aid = store.add(
            label=label, at=at, asked_by=who, obs_id=obs_id, passes_quiet=bool(reason), now=now
        )
        logger.info(
            "アラームを掛けた id=%d %s at=%s 通り抜け=%s", aid, label, at.isoformat(), bool(reason)
        )
        return f"掛けた：id={aid} 「{label}」 {at:%H:%M}", True

    async def _cancel(self, inp: dict) -> tuple[str, bool]:
        if self._on_cancel is not None:
            self._on_cancel()
        now = self._now()
        store = self._store()
        target = inp.get("id")
        if str(target).strip().lower() == "all":
            rows = store.active(now=now)
            if not rows:
                return "掛かっているアラームは無い", True
            n = store.cancel_all(now=now)
            await self._write("やめた：" + "・".join(f"「{r['label']}」のアラーム" for r in rows))
            logger.info("アラームを全部止めた（%d 本）", n)
            return f"{n} 本止めた：" + "・".join(f"「{r['label']}」" for r in rows), True
        try:
            aid = int(str(target))
        except (TypeError, ValueError):
            return f"id を読めない：{target}", False
        row = next((r for r in store.active(now=now) if int(r["id"]) == aid), None)
        if row is None or not store.cancel(aid, now=now):
            return f"id={aid} のアラームは掛かっていない", False
        await self._write(
            f"やめた：「{row['label']}」のアラーム（{row['at'].astimezone():%H:%M} の予定だった）"
        )
        logger.info("アラームを止めた id=%d %s", aid, row["label"])
        return f"止めた：id={aid} 「{row['label']}」", True

    async def _write(self, content: str) -> "str | None":
        try:
            return await self._oif.write(
                MI(id="", content=content[:500], timestamp=None, direction="予定"),
                writer_id=AGENT_SELF_ID,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("予定の記録を書けなかった: %s", e)
            return None
