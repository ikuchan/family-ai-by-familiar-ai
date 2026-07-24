"""動体検知（ONVIF PullPoint）→ 知覚ターン起動のバックグラウンド購読（案B）。

`CameraPresenceWatcher` と同型の asyncio タスク。カメラの ONVIF イベントを PullPoint で
購読し、動体を検知したら `on_motion` コールバックを呼ぶ（デバウンスで連発を1回にまとめる）。
DIF（純イベント駆動 I）は未実装なので、当面は現行ターン駆動へ接地する（将来 DIF へ載せ替え）。

動作手順（実機で確認済み）：
  create_pullpoint_manager(timedelta, lost_cb) → mgr.get_service()
  → SetSynchronizationPoint() → ループで PullMessages({Timeout, MessageLimit})
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..config import CameraConfig

logger = logging.getLogger(__name__)

# motion とみなすトピックの語（ONVIF の RuleEngine/CellMotionDetector/Motion 等）。
_MOTION_TOKENS = ("motion", "motionalarm")
# アクティブ状態を表す SimpleItem 名。
_ACTIVE_KEYS = ("ismotion", "state", "motion", "isinside")


def is_motion_active(topic: str, items: "dict[str, str]") -> bool:
    """通知が「動体あり」か。motion トピックで、状態 item が true（または item 無し）なら True。"""
    if "motion" not in (topic or "").lower():
        return False
    if not items:
        return True  # motion トピックだけ（明示 item なし）＝発生とみなす
    for k, v in items.items():
        if k.lower() in _ACTIVE_KEYS and str(v).strip().lower() in ("true", "1"):
            return True
    return False


class Debouncer:
    """一度発火したら window_sec は再発火しない（バーストを1回にまとめる）。"""

    def __init__(self, window_sec: float) -> None:
        self._window = window_sec
        self._last: float | None = None

    def allow(self, now: float) -> bool:
        if self._last is None or (now - self._last) >= self._window:
            self._last = now
            return True
        return False


def _message_to_topic_items(msg: Any) -> "tuple[str, dict[str, str]]":
    """zeep の NotificationMessage から (topic, {name: value}) を防御的に取り出す。"""
    topic = ""
    try:
        t = getattr(msg, "Topic", None)
        topic = str(getattr(t, "_value_1", "") or "")
    except Exception:  # noqa: BLE001
        topic = ""
    items: dict[str, str] = {}
    try:
        message = getattr(msg, "Message", None)
        inner = getattr(message, "_value_1", None) if message is not None else None
        data = getattr(inner, "Data", None) if inner is not None else None
        simple = getattr(data, "SimpleItem", None) if data is not None else None
        for si in simple or []:
            name = getattr(si, "Name", None)
            value = getattr(si, "Value", None)
            if name is not None:
                items[str(name)] = str(value)
    except Exception:  # noqa: BLE001
        pass
    return topic, items


def notifications_have_motion(messages: Any) -> bool:
    """PullMessages の NotificationMessage 群に動体ありが含まれるか。"""
    for msg in messages or []:
        topic, items = _message_to_topic_items(msg)
        if is_motion_active(topic, items):
            return True
    return False


class CameraMotionWatcher:
    """ONVIF PullPoint で動体を購読し、検知で on_motion を呼ぶ（デバウンス付き・degrade 対応）。"""

    def __init__(
        self,
        camera: "CameraConfig",
        on_motion: Callable[[], None],
        *,
        pull_timeout_sec: float = 60.0,
        debounce_sec: float = 60.0,
    ) -> None:
        self._camera = camera
        self._on_motion = on_motion
        self._pull_timeout = pull_timeout_sec
        self._debounce = Debouncer(debounce_sec)
        self._task: asyncio.Task | None = None
        self._closing = False

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._closing = False
        self._task = asyncio.create_task(self._loop(), name="motion-watcher")
        logger.info("CameraMotionWatcher started (pull=%.0fs)", self._pull_timeout)

    async def stop(self) -> None:
        self._closing = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _connect(self) -> Any:
        """ONVIF に繋ぎ、SetSynchronizationPoint 済みの PullPoint (manager, service) を返す。"""
        from onvif import ONVIFCamera

        from ..setup import _onvif_wsdl_dir

        cam = ONVIFCamera(
            self._camera.ptz_host or self._camera.host,
            self._camera.ptz_port or self._camera.port,
            self._camera.username,
            self._camera.password,
            wsdl_dir=_onvif_wsdl_dir(),
        )
        await cam.update_xaddrs()
        mgr = await cam.create_pullpoint_manager(
            timedelta(seconds=self._pull_timeout), lambda: None
        )
        pp = mgr.get_service()
        await pp.SetSynchronizationPoint()
        return cam, mgr, pp

    async def _loop(self) -> None:
        backoff = 2.0
        while not self._closing:
            cam = mgr = None
            try:
                cam, mgr, pp = await self._connect()
                backoff = 2.0  # 接続できたら backoff リセット
                while not self._closing:
                    msgs = await pp.PullMessages(
                        {"Timeout": timedelta(seconds=self._pull_timeout), "MessageLimit": 10}
                    )
                    nm = getattr(msgs, "NotificationMessage", None)
                    if notifications_have_motion(nm):
                        loop = asyncio.get_event_loop()
                        if self._debounce.allow(loop.time()):
                            logger.info("Motion detected → 知覚ターン起動を要求")
                            try:
                                self._on_motion()
                            except Exception as e:  # noqa: BLE001
                                logger.warning("on_motion callback error: %s", e)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning("MotionWatcher error (retry in %.0fs): %s", backoff, e)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
            finally:
                if mgr is not None:
                    with contextlib.suppress(Exception):
                        await mgr.shutdown()
                if cam is not None:
                    with contextlib.suppress(Exception):
                        await cam.close()
