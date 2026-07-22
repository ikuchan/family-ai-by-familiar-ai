"""Background camera poller for passive presence detection.

Runs as an asyncio task. Polls the camera every `interval_sec` seconds,
runs face recognition, and updates PersonMemoryManager accordingly.
Persons not seen for `absent_threshold_sec` are marked as left.
"""
from __future__ import annotations
import asyncio, base64, logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..person_memory_manager import PersonMemoryManager
    from ..config import CameraConfig

logger = logging.getLogger(__name__)


def _encode_frame(path: str) -> str | None:
    """フレーム画像ファイルを base64 文字列にする（GUI 表示用）。読めなければ None。"""
    try:
        return base64.b64encode(Path(path).read_bytes()).decode("ascii")
    except Exception:
        return None


class CameraPresenceWatcher:
    def __init__(
        self,
        manager: "PersonMemoryManager",
        camera: "CameraConfig | None" = None,
        interval_sec: float = 5.0,
        absent_threshold_sec: float = 30.0,
    ) -> None:
        self._manager  = manager
        from ..config import CameraConfig as _CameraConfig
        self._camera   = camera or _CameraConfig()
        self._interval = interval_sec
        self._absent   = absent_threshold_sec
        self._task: asyncio.Task | None = None
        # 直近に認識用へ撮ったフレーム（base64）。GUI が在席確認カメラとして表示する。
        self._last_frame_b64: str | None = None

    def latest_frame_b64(self) -> str | None:
        """認識に使った直近フレームの base64（まだ無ければ None）。"""
        return self._last_frame_b64

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
                    self._last_frame_b64 = _encode_frame(frame_path)
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

    async def _capture_frame(self) -> str | None:
        """Capture one frame from the camera. Returns temp file path or None."""
        try:
            import os
            import tempfile

            cam = self._camera
            if cam.is_rtsp():
                url = cam.stream_url("stream1")
                tmp = tempfile.mktemp(suffix=".jpg")
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-y", "-rtsp_transport", "tcp",
                    "-i", str(url), "-frames:v", "1", "-q:v", "3", tmp,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(proc.wait(), timeout=8.0)
                if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                    return tmp
                # RTSP failed — do NOT fall through to USB when RTSP is configured.
                # Falling through would open /dev/video0 in an infinite error loop.
                return None
            # USB webcam (only when CAMERA_HOST is not set)
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
