"""Camera tool - the eyes and neck of the embodied agent.

Built-in tools:
- see(): capture a camera frame and describe the scene via vision LLM.
- look(direction, degrees): rotate the camera head via ONVIF PTZ.
  direction: left/right/up/down. Supported on pan-tilt cameras (e.g. Tapo C211).
  PTZ availability is auto-detected at startup via is_pan_tilt_available().
Config: CAMERA_HOST, CAMERA_USERNAME, CAMERA_PASSWORD, CAMERA_ONVIF_PORT (default 2020).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
from onvif import ONVIFCamera

from ..poses import Pose
from ..setup import _onvif_wsdl_dir

logger = logging.getLogger(__name__)

CAPTURE_DIR = Path.home() / ".familiar_ai" / "captures"


class CameraTool:
    """Controls a camera via OpenCV (RTSP, USB, file) and optionally via ONVIF (PTZ)."""

    def __init__(
        self,
        host: str | int,
        username: str | None = None,
        password: str | None = None,
        port: int = 2020,
        preview: bool = False,
        *,
        ptz_host: str | int | None = None,
        ptz_username: str | None = None,
        ptz_password: str | None = None,
        ptz_port: int | None = None,
    ):
        self.host = host
        self.username = username
        self.password = password
        self.port = port
        self.preview = preview
        self.ptz_host = ptz_host if ptz_host is not None else host
        self.ptz_username = ptz_username
        self.ptz_password = ptz_password
        self.ptz_port = ptz_port if ptz_port is not None else port

        self._cam_onvif: Any = None
        self._ptz: Any = None
        self._profile_token: str | None = None

        self._cap: cv2.VideoCapture | None = None
        self._last_frame: Any = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Start background capture thread
        self.start()

    @property
    def is_pan_tilt_available(self) -> bool:
        """Check if PTZ controls are supported by the camera."""
        hostname, _username, _password, _port = self._get_ptz_connection_params()
        # ONVIF is only attempted if PTZ host is not a simple integer (USB)
        if hostname is None:
            return False
        if isinstance(hostname, int) or (isinstance(hostname, str) and hostname.isdigit()):
            return False
        # If PTZ service is already connected, it's available.
        # Otherwise, assume it's available if it's an IP camera (will be lazy-connected later).
        return True

    def start(self):
        """Start the background capture thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def close(self):
        """Stop capture and release resources."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera resources released.")

    def _capture_loop(self):
        """Background thread to keep camera buffer fresh and optionally show preview."""
        source = self._get_stream_url()

        # Suppress ffmpeg C-level warnings (SEI type 764 spam from Tapo).
        # Cannot redirect stderr (breaks Textual TUI), so use OPENCV env vars instead.
        os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # AV_LOG_QUIET
        os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

        def _open_cap() -> cv2.VideoCapture:
            if isinstance(source, str) and source.startswith("rtsp://"):
                # stimeout: socket-level read timeout in microseconds (5 s).
                # Reduces dead-stream recovery from 30–47 s to ~5 s.
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    "rtsp_transport;tcp|stimeout;5000000"
                )
            return cv2.VideoCapture(
                source, cv2.CAP_FFMPEG if isinstance(source, str) else cv2.CAP_ANY
            )

        try:
            self._cap = _open_cap()

            if not self._cap.isOpened():
                logger.error("Failed to open camera source: %s", source)
                self._running = False
                return

            logger.info("Camera capture thread started for source: %s", source)

            while self._running:
                ret, frame = self._cap.read()
                if not ret:
                    logger.warning("Failed to read frame, retrying in 2s...")
                    self._cap.release()
                    self._cap = None
                    time.sleep(2.0)
                    self._cap = _open_cap()
                    continue

                with self._lock:
                    self._last_frame = frame.copy()

                if self.preview:
                    cv2.imshow("Familiar-AI Camera Preview", frame)
                    # waitKey is required for imshow to actually render
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.preview = False
                        cv2.destroyAllWindows()

            if self._cap:
                self._cap.release()
                self._cap = None
        finally:
            pass

    async def _ensure_connected(self) -> bool:
        """Ensure ONVIF connection is established for PTZ (optional)."""
        if self._cam_onvif is not None:
            return True

        hostname, username, password, port = self._get_ptz_connection_params()
        if hostname is None:
            return False
        if isinstance(hostname, int) or (isinstance(hostname, str) and hostname.isdigit()):
            return False

        wsdl_dir = _onvif_wsdl_dir()

        # Try the configured port first, then common fallback ports (Eufy/generic cameras
        # often use 8080 or 80 instead of Tapo's default 2020).
        ports_to_try = [port]
        for fallback in (8080, 80):
            if fallback != port:
                ports_to_try.append(fallback)

        last_error: Exception | None = None
        for try_port in ports_to_try:
            cam = None
            try:
                cam = ONVIFCamera(hostname, try_port, username, password, wsdl_dir=wsdl_dir)
                await cam.update_xaddrs()
                media = await cam.create_media_service()
                profiles = await media.GetProfiles()
                self._profile_token = profiles[0].token if profiles else "Profile_1"
                self._ptz = await cam.create_ptz_service()
                self._cam_onvif = cam
                logger.info("Camera PTZ connected via ONVIF: %s (port %d)", hostname, try_port)
                return True
            except Exception as e:
                logger.debug("ONVIF PTZ port %d failed for %s: %s", try_port, hostname, e)
                last_error = e
                # 失敗した接続のセッションを閉じる（ポート総当りで漏れが積もるのを防ぐ）。
                if cam is not None:
                    with contextlib.suppress(Exception):
                        await cam.close()

        logger.warning(
            "ONVIF PTZ unavailable for %s (tried ports %s). "
            "Pan/tilt will be disabled. Last error: %s. "
            "Tip: set CAMERA_PTZ_PORT to the correct ONVIF port for your camera "
            "(Tapo=2020, Eufy=8080).",
            hostname,
            ports_to_try,
            last_error,
        )
        return False

    def _get_stream_url_parts(self):
        if isinstance(self.host, str) and "://" in self.host:
            return urlparse(self.host)
        return None

    def _get_ptz_connection_params(
        self,
    ) -> tuple[str | int | None, str | None, str | None, int]:
        ptz_host = self.ptz_host
        if isinstance(ptz_host, str) and "://" in ptz_host:
            parsed_ptz = urlparse(ptz_host)
            hostname: str | int | None = parsed_ptz.hostname or ptz_host
        else:
            hostname = ptz_host

        stream_parts = self._get_stream_url_parts()
        stream_username = stream_parts.username if stream_parts is not None else None
        stream_password = stream_parts.password if stream_parts is not None else None

        username = self.ptz_username or self.username or stream_username
        password = self.ptz_password or self.password or stream_password
        port = self.ptz_port
        return hostname, username, password, port

    def _get_stream_url(self) -> str | int:
        from ..config import CameraConfig
        return CameraConfig(
            host=str(self.host) if self.host is not None else "",
            username=self.username or "",
            password=self.password or "",
        ).stream_url("stream1")

    async def capture(self) -> tuple[str | None, str | None]:
        """Get the latest frame from the background thread. Returns (base64_jpeg, saved_path)."""
        frame = None
        with self._lock:
            if self._last_frame is not None:
                frame = self._last_frame.copy()

        if frame is None:
            logger.warning("No frame available from capture thread.")
            return None, None

        try:
            # Resize for AI (standardizing input size)
            h, w = frame.shape[:2]
            target_h = 640
            if h > target_h:
                scale = target_h / h
                frame = cv2.resize(frame, (int(w * scale), target_h))

            # Encode
            success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                return None, None

            data = buffer.tobytes()
            b64 = base64.b64encode(data).decode()

            # Save to disk
            CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = CAPTURE_DIR / f"capture_{timestamp}.jpg"
            save_path.write_bytes(data)

            return b64, str(save_path)
        except Exception as e:
            logger.warning("Error processing frame: %s", e)
            return None, None

    async def move_to(self, pan: float, tilt: float) -> str:
        """定点へ向く（絶対 pan/tilt）。

        見回りは「絶対 pan/tilt で1定点へ移動 → 観察」を1ステップとする（`ユースケース③`）。
        相対移動では、誤差が積もって定点からずれていく。範囲外は手前で丸める（カメラに
        拒否させるより、届く範囲まで動かすほうがよい）。
        """
        if not await self._ensure_connected():
            return "Camera movement (PTZ) not supported for this source."
        pan = max(-1.0, min(1.0, float(pan)))
        tilt = max(-1.0, min(1.0, float(tilt)))
        try:
            await self._ptz.AbsoluteMove(
                {
                    "ProfileToken": self._profile_token,
                    "Position": {"PanTilt": {"x": pan, "y": tilt}},
                }
            )
            await asyncio.sleep(0.4)
            return f"Turned to pan={pan:+.4f} tilt={tilt:+.4f}."
        except Exception as e:  # noqa: BLE001
            logger.warning("Camera absolute move failed: %s", e)
            return f"Camera move failed: {e}"

    async def position(self) -> tuple[float, float] | None:
        """いまどこを向いているか。読めなければ `None`。

        分からないときに (0,0) を返すと、向いていない定点を向いていることにしてしまい、
        その定点の「普通」と在席マップが別の場所の映像で汚れる。
        """
        if not await self._ensure_connected():
            return None
        try:
            status = await self._ptz.GetStatus({"ProfileToken": self._profile_token})
            pt = status.Position.PanTilt
            return (float(pt.x), float(pt.y))
        except Exception as e:  # noqa: BLE001
            logger.warning("Camera position read failed: %s", e)
            return None

    async def presets(self) -> list[Pose]:
        """カメラに登録されたプリセットを定点として読む。

        人がカメラのアプリで足したものがそのまま定点になる。読めなければ空を返す
        （Config の定点だけで動ける）。
        """
        if not await self._ensure_connected():
            return []
        try:
            raw = await self._ptz.GetPresets({"ProfileToken": self._profile_token})
        except Exception as e:  # noqa: BLE001
            logger.warning("Camera preset read failed: %s", e)
            return []
        out: list[Pose] = []
        for item in raw or []:
            pos = getattr(item, "PTZPosition", None)
            pt = getattr(pos, "PanTilt", None) if pos is not None else None
            if pt is None:
                continue
            name = str(getattr(item, "Name", None) or getattr(item, "token", ""))
            out.append(Pose(name, float(pt.x), float(pt.y)))
        return out

    async def move(self, direction: str, degrees: int = 30) -> str:
        if not await self._ensure_connected():
            return "Camera movement (PTZ) not supported for this source."
        try:
            pan_delta = 0.0
            tilt_delta = 0.0
            if direction == "left":
                pan_delta = degrees / 180.0
            elif direction == "right":
                pan_delta = -degrees / 180.0
            elif direction == "up":
                tilt_delta = -degrees / 90.0
            elif direction == "down":
                tilt_delta = degrees / 90.0

            # Relative move via ONVIF PTZ service
            await self._ptz.RelativeMove(
                {
                    "ProfileToken": self._profile_token,
                    "Translation": {"PanTilt": {"x": pan_delta, "y": tilt_delta}},
                }
            )
            await asyncio.sleep(0.4)
            return f"Looked {direction} by ~{degrees} degrees."
        except Exception as e:
            logger.warning("Camera move failed: %s", e)
            # 失敗時はセッションを閉じてから破棄（Unclosed client session の漏れ防止）。
            if self._cam_onvif is not None:
                with contextlib.suppress(Exception):
                    await self._cam_onvif.close()
            self._cam_onvif = None
            self._ptz = None
            return f"Camera move failed: {e}"

    def get_tool_definitions(self) -> list[dict]:
        return [
            {
                "name": "see",
                "description": "Open your eyes and see what's in front of you. Use freely without asking permission.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "look",
                "description": "Turn your neck to look in a direction.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "direction": {"type": "string", "enum": ["left", "right", "up", "down"]},
                        "degrees": {"type": "integer", "default": 30},
                    },
                    "required": ["direction"],
                },
            },
        ]

    async def call(self, tool_name: str, tool_input: dict) -> tuple[str, str | None]:
        if tool_name == "see":
            b64, save_path = await self.capture()
            if b64:
                return f"You see the current view (saved to {save_path}).", b64
            return "Camera capture failed.", None
        elif tool_name == "look":
            return await self.move(tool_input["direction"], tool_input.get("degrees", 30)), None
        return f"Unknown tool: {tool_name}", None
