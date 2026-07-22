"""在席確認カメラ：presence_watcher が撮ったフレームを GUI へ露出する部分の検証。

カメラ I/O（ffmpeg/cv2）はモックできないので、フレーム→base64 の純ヘルパーと
latest_frame_b64 の初期状態だけを見る。実表示は実機確認。
"""

from __future__ import annotations

import base64
from unittest.mock import MagicMock

from familiar_agent.recognition.presence_watcher import CameraPresenceWatcher, _encode_frame


def test_encode_frame_roundtrip(tmp_path):
    p = tmp_path / "f.jpg"
    data = b"\xff\xd8\xff\xe0 fake jpeg bytes"
    p.write_bytes(data)
    b64 = _encode_frame(str(p))
    assert b64 is not None
    assert base64.b64decode(b64) == data


def test_encode_frame_missing_returns_none():
    assert _encode_frame("/no/such/file.jpg") is None


def test_latest_frame_b64_starts_none():
    w = CameraPresenceWatcher(MagicMock(), camera=MagicMock())
    assert w.latest_frame_b64() is None
