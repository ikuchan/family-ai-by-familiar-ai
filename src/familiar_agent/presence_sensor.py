"""在/不在を測る器：撮って、どの定点かを決めて、人が居るかを数える。

`知覚在席` §3-2 の在/不在は **G（T 側・連続）** が担う。ここが部品を束ねる場所で、定点
（`poses`）、人検出（`recognition.person_detector`）、定点別の記録（`presence_map`）、
動体イベント（`recognition.motion_events`）を繋ぐ。**誰かは問わない**（#17 の担当）。

起こされ方は2つある。カメラが「動いた」と言ってきたときと、一定間隔（既定30秒）。動体
だけでは足りない。静止している人は動体を出さないので、動きが無いあいだも確かめる。

**どの定点を見ているか分からないときは、何も記録しない**（`知覚在席` §3-3 の振動中ゲート）。
向きが読めない、または定点から離れているときの映像は、どの定点のものでもない。撮れなかった
ときも同じで、「誰も居ない」にはしない。カメラの不調で人が消えることになる。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from typing import Any

from .poses import nearest_pose
from .presence_map import PresenceMap

logger = logging.getLogger(__name__)


def _delay_before(last_check: float, now: float, min_gap: float) -> float:
    """次の確認まで待つ秒数。前回から間が空いていれば 0（すぐ確かめる）。

    動いているあいだ動体イベントは毎秒何件も飛ぶ。そのたびに撮って YOLO を回すと、実機では
    **0.15 秒ごと**に確認していた。動き始めには即座に気づきたいので最初の1件は待たせず、
    続けて飛んでくるぶんだけ間引く。
    """
    return max(0.0, min_gap - (now - last_check))


class PresenceSensor:
    """定点ごとに人が居るかを見て、`PresenceMap` を更新し続ける。"""

    def __init__(self, camera: Any, poses_getter: Callable[[], Any], detector: Any,
                 *, tolerance: float, window_sec: float, interval_sec: float,
                 min_gap_sec: float = 3.0) -> None:
        self._camera = camera
        self._poses_getter = poses_getter
        self._detector = detector
        self._tolerance = tolerance
        self._window = window_sec
        self._interval = interval_sec
        self._min_gap = min_gap_sec
        self._last_check = float("-inf")
        self._map: PresenceMap | None = None
        self._frame_b64: str | None = None
        self._task: asyncio.Task | None = None
        # 動体イベントで即座に起こすための合図。間隔の待ちを飛び越える。
        self._wake = asyncio.Event()

    # --- 外から読むもの ---------------------------------------------------

    def room_occupied(self) -> bool:
        return self._map.room_occupied(time.time()) if self._map else False

    def poses_seen(self) -> list[str]:
        return self._map.poses_seen(time.time()) if self._map else []

    def stalest_pose(self) -> str | None:
        """次に見に行くべき定点。見回り（S5）が使う。"""
        return self._map.stalest_pose(time.time()) if self._map else None

    def latest_frame_b64(self) -> str | None:
        """直近に見たフレーム。GUI が在席確認のカメラ映像として表示する。"""
        return self._frame_b64

    def on_motion(self) -> None:
        """カメラが「動いた」と言ってきた。次の間隔を待たずに確かめる。"""
        self._wake.set()

    # --- 中身 -------------------------------------------------------------

    async def _ensure_map(self) -> PresenceMap | None:
        if self._map is None:
            poses = await self._poses_getter()
            if not poses:
                return None
            self._map = PresenceMap([p.name for p in poses], window_sec=self._window)
            self._poses = poses
        return self._map

    async def check_once(self) -> str | None:
        """1回見る。記録した定点の名前を返す。記録しなかったときは `None`。"""
        try:
            pmap = await self._ensure_map()
            if pmap is None:
                return None
            position = await self._camera.position()
            if position is None:
                logger.debug("向きが読めないので在席を記録しない")
                return None
            pose = nearest_pose(self._poses, position[0], position[1], self._tolerance)
            if pose is None:
                logger.debug("どの定点でもない向きなので在席を記録しない（移動中）")
                return None
            frame_b64, path = await self._camera.capture()
            if not path:
                logger.warning("フレームを撮れなかったので在席を記録しない")
                return None
            self._frame_b64 = frame_b64
            people = await self._detector.count(path)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("在席の確認に失敗した（記録しない）: %s", e)
            return None
        now = time.time()
        if people > 0:
            pmap.mark_seen(pose.name, now)
            logger.info("在席：%s に %d 人", pose.name, people)
        else:
            pmap.mark_checked(pose.name, now)
            logger.debug("在席：%s に誰も居ない", pose.name)
        return pose.name

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="presence-sensor")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _run(self) -> None:
        while True:
            await self.check_once()
            self._last_check = time.monotonic()
            self._wake.clear()
            try:
                # 動体で起こされたら間隔を待たずに次を見る。
                await asyncio.wait_for(self._wake.wait(), timeout=self._interval)
            except (TimeoutError, asyncio.TimeoutError):
                continue                      # 動きが無いまま間隔が来た
            delay = _delay_before(self._last_check, time.monotonic(), self._min_gap)
            if delay > 0:
                await asyncio.sleep(delay)
