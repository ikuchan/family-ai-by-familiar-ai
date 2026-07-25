"""T（自律機構 Tonic）の常駐タスク：時計を持つ唯一の側。

正本＝`設計図_Mermaid` ③・課題5 A節。役割分担は次のとおり。

- **T（ここ）**：時計を見る。$P_T$ ごとに drive を蓄積し、閾値で発火させ、**AIF 経由で QA へ積む**。
- **I（駆動体）**：時計を見ない。3キューを待ち、来たどれでも起きる。

drive の蓄積と発火判定は `core.drive_dynamics` の純関数が持ち、ここは時間で回して永続化し、
発火を QA へ渡すだけにする。同じ処理は GUI の描画ループにも埋まっていた（`gui.py:_tick_drives`）
一方で CUI には無く、自律が動く条件が入口ごとに違っていた。T を1本立てて揃える。
"""

from __future__ import annotations

import asyncio
import logging
import time

from ..config import DriveConfig
from ..core import drive_dynamics as dd
from ..core.drive_autonomy import inner_voice_for, select_fired_axis
from ..drive_register import AiDrivers, load_drives, save_drives
from ..mood_register import load_current_mood

logger = logging.getLogger(__name__)

# 課題5 A節の確定値。蓄積式 drive_i ← clip(drive_i + rate·mult·learn·g_D·P_T) にも入るので、
# 周期を変えると蓄積そのものが変わる。勝手に動かさない。
TONIC_PERIOD_SEC = 0.5


async def step_drives(dt: float) -> tuple[dd.DriveFiring, AiDrivers]:
    """1 tick 分の dynamics を回して永続化し、(発火, 蓄積後・放電前の drives) を返す。

    重い呼び出しではないが DB を触るのでスレッドへ逃がす。`load_current_mood` は内部で
    `db.lock` を取り再入できないため、ロックを取る前に読む（既存 GUI 実装と同じ順序）。
    """
    def _work() -> tuple[dd.DriveFiring, AiDrivers]:
        from ..db import get_db

        cfg = DriveConfig()
        mood = load_current_mood()          # 自己接続でロックを取り、抜ける
        database = get_db()
        with database.lock:
            conn = database.conn()
            drives = load_drives(conn)
            accumulated = dd.accumulate(drives, mood, dt=dt, cfg=cfg)
            firing = dd.fired(accumulated, cfg)
            persisted = dd.discharge(accumulated, firing, cfg) if firing.any else accumulated
            save_drives(conn, persisted)
        return firing, accumulated

    return await asyncio.to_thread(_work)


class Tonic:
    """自律機構の常駐タスク。$P_T$ ごとに drive を進め、発火を QA へ積む。"""

    def __init__(self, information_processing, *, period: float = TONIC_PERIOD_SEC) -> None:
        self._ip = information_processing
        self._period = period
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    async def _run(self) -> None:
        last = time.monotonic()
        while True:
            try:
                await asyncio.sleep(self._period)
                now = time.monotonic()
                dt, last = now - last, now
                firing, accumulated = await step_drives(dt)
                if not firing.any:
                    continue
                axis = select_fired_axis(firing, accumulated)
                if axis is None:
                    continue
                prompt = inner_voice_for(axis, DriveConfig())
                logger.info("Drive fired: %s → QA へ積む", axis)
                self._ip.push_affect(axis.upper(), prompt)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                # 自律は落とさない（アイドルが止まると何も起きなくなる）。
                logger.exception("tonic tick に失敗: %s", e)
