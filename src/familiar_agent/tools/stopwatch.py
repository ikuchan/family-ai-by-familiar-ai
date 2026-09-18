"""ストップウォッチの道具（知-u・2026-09-18・`設計方針_ストップウォッチ` v0.1）。タイマー（`tools/timer.py`）とは別物。

主LLM が呼ぶ 2 本——`start_stopwatch`（今から測る）・`stop_stopwatch`（止めて経過を答える）。調停の候補にも
同じ 2 本。鳴らない・確認しない・黙らない・聞かない状態にしない・一時停止なし・同時 1 本。寿命は
`STOPWATCH_MAX_SEC`（6 時間）で T が自動で止める（`loop/stopwatch_watch.py`）。共有するのは `予定` の記録だけ。
09-16 に主LLM が始めた 2 本が 2 日間動き続け、`/timer stop` の返事「2 本止めた」で何を止めたか分からなかった。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from ..core import stopwatch_rules
from ..io.oif import MI
from ..person_memory_manager import AGENT_SELF_ID

logger = logging.getLogger(__name__)

MAX_ACTIVE = 1  # 同時に 1 本（タイマーとは別枠）
RECENT_SEC = 180.0  # 止めた後も枠に残す秒数〔仮〕（「何分だった？」に答える）

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "start_stopwatch",
        "description": (
            "ストップウォッチを始める（「今から測って」「何分かかるか測って」）。鳴らない。経過は [ストップウォッチ] の枠で分かる。"
            "同時に 1 本。何分後に鳴らすならタイマー（set_timer）。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "何を測るか（短く・例「お風呂」「散歩」）",
                }
            },
            "required": ["label"],
        },
    },
    {
        "name": "stop_stopwatch",
        "description": (
            "ストップウォッチを止めて、測った長さを答える（「ストップ」「もういいよ」「何分だった？」）。"
            'id は [ストップウォッチ] の枠の番号。全部止めるなら "all"。'
        ),
        "input_schema": {
            "type": "object",
            "properties": {"id": {"description": '番号か "all"'}},
            "required": ["id"],
        },
    },
]


class StopwatchTool:
    def __init__(
        self,
        *,
        store: Callable[[], Any],
        oif: Any,
        speaker: Callable[[], str],
        max_sec: float,
        now: "Callable[[], datetime] | None" = None,
    ) -> None:
        self._store = store
        self._oif = oif
        self._speaker = speaker
        self._max_sec = float(max_sec)
        self._now = now or (lambda: datetime.now(timezone.utc).astimezone())

    def store(self):
        return self._store()

    def get_tool_definitions(self) -> list[dict]:
        return [dict(d) for d in TOOL_DEFINITIONS]

    def frame(self) -> str:
        """`[ストップウォッチ]` の枠。無ければ空。"""
        now = self._now()
        store = self._store()
        return stopwatch_rules.render_frame(
            store.active(now=now),
            store.recently_stopped(now=now, within_sec=RECENT_SEC),
            now=now,
            max_sec=self._max_sec,
        )

    async def call(
        self, name: str, tool_input: dict, *, now: "datetime | None" = None
    ) -> tuple[str, bool]:
        """(文面, 使えたか)。`now` は起点（人が言った瞬間・無ければいま）。"""
        try:
            if name == "start_stopwatch":
                return await self._start(tool_input, now=now)
            if name == "stop_stopwatch":
                return await self._stop(tool_input, now=now)
        except Exception as e:  # noqa: BLE001
            logger.exception("ストップウォッチの道具に失敗: %s", e)
            return f"ストップウォッチの道具が使えなかった：{e}", False
        return f"そんな道具は無い：{name}", False

    async def _start(self, inp: dict, *, now: "datetime | None") -> tuple[str, bool]:
        label = str(inp.get("label") or "測る").strip()
        now = (now or self._now()).astimezone()
        store = self._store()
        running = store.active(now=now)
        if len(running) >= MAX_ACTIVE:
            r = running[0]
            run = stopwatch_rules.elapsed(r, now).total_seconds()
            return (
                f"いま「{r['label']}」（id={r['id']}・経過 {stopwatch_rules.elapsed_text(run)}）を測っている。止めてから",
                False,
            )
        who = self._speaker() or ""
        obs_id = await self._write(
            f"{who or '誰か'}に頼まれて、{now:%H:%M} から「{label}」を測り始めた"
        )
        wid = store.add(label=label, asked_by=who, obs_id=obs_id, now=now)
        logger.info("ストップウォッチを始めた id=%d %s", wid, label)
        return f"測り始めた：id={wid} 「{label}」 {now:%H:%M} から", True

    async def _stop(self, inp: dict, *, now: "datetime | None") -> tuple[str, bool]:
        now = (now or self._now()).astimezone()
        store = self._store()
        target = inp.get("id")
        rows = store.active(now=now)
        if str(target).strip().lower() == "all":
            if not rows:
                return "動いているストップウォッチは無い", True
            for r in rows:
                store.stop(int(r["id"]), now=now)
            logger.info("ストップウォッチを全部止めた（%d 本）", len(rows))
            return "・".join(self._stopped_text(r, now) for r in rows), True
        try:
            wid = int(str(target).strip())
        except (TypeError, ValueError):
            return f"id を読めない：{target}", False
        row = next((r for r in rows if int(r["id"]) == wid), None)
        if row is None or not store.stop(wid, now=now):
            return f"id={wid} のストップウォッチは動いていない", False
        logger.info("ストップウォッチを止めた id=%d %s", wid, row["label"])
        return self._stopped_text(row, now), True

    def _stopped_text(self, row: dict, now: datetime) -> str:
        run = stopwatch_rules.elapsed(row, now).total_seconds()
        return f"ストップウォッチ「{row['label']}」を止めた（{stopwatch_rules.elapsed_text(run)} 経過）"

    async def _write(self, content: str) -> "str | None":
        try:
            return await self._oif.write(
                MI(id="", content=content[:500], timestamp=None, direction="予定"),
                writer_id=AGENT_SELF_ID,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("予定の記録を書けなかった: %s", e)
            return None
