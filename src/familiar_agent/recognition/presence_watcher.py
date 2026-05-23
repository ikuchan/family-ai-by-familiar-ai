"""Background camera poller for passive presence detection.

Runs as an asyncio task. Polls the camera every `interval_sec` seconds,
runs face recognition, and updates PersonMemoryManager accordingly.
Persons not seen for `absent_threshold_sec` are marked as left.
"""
from __future__ import annotations
import asyncio, logging, time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..person_memory_manager import PersonMemoryManager

logger = logging.getLogger(__name__)


class CameraPresenceWatcher:
    def __init__(
        self,
        manager: "PersonMemoryManager",
        interval_sec: float = 5.0,
        absent_threshold_sec: float = 30.0,
    ) -> None:
        self._manager  = manager
        self._interval = interval_sec
        self._absent   = absent_threshold_sec
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop(), name="presence-watcher")
        logger.info("CameraPresenceWatcher started (interval=%.1fs)", self._interval)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        from .face import recognize_face_async
        while True:
            try:
                frame_path = await self._capture_frame()
                if frame_path:
                    hint = await recognize_face_async(frame_path, self._manager)
                    if hint:
                        await self._manager.apply_hint(hint)

                # Mark stale persons as left
                for pid in self._manager.stale_present_ids():
                    logger.info("Presence timeout for %s — marking left", pid)
                    await self._manager.person_left(pid)
            except Exception as e:
                logger.warning("PresenceWatcher error: %s", e)
            await asyncio.sleep(self._interval)

    @staticmethod
    async def _capture_frame() -> str | None:
        """Capture one frame from the camera. Returns temp file path or None."""
        try:
            import tempfile, os
            # Try RTSP first (Tapo), fall back to USB webcam
            rtsp = os.environ.get("CAMERA_HOST")
            if rtsp:
                user = os.environ.get("CAMERA_USER", "")
                pw   = os.environ.get("CAMERA_PASS", "")
                url  = f"rtsp://{user}:{pw}@{rtsp}/stream1" if user else f"rtsp://{rtsp}/stream1"
                tmp  = tempfile.mktemp(suffix=".jpg")
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-y", "-rtsp_transport", "tcp",
                    "-i", url, "-frames:v", "1", "-q:v", "3", tmp,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(proc.wait(), timeout=8.0)
                if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                    return tmp
            # USB webcam via OpenCV
            import cv2
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return None
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None
            tmp = tempfile.mktemp(suffix=".jpg")
            cv2.imwrite(tmp, frame)
            return tmp
        except Exception as e:
            logger.debug("Frame capture error: %s", e)
            return None
