"""**カメラのログに認証情報を出さない**（2026-09-12 実機で露見）。

`rtsp://user:pass@host/...` をそのままログへ出しており、パスワードが `app.log` と画面に
残っていた。ホストと経路は切り分けに要るので残し、認証情報だけ伏せる。
"""

from __future__ import annotations

import inspect

from familiar_agent.tools import camera


def test_credentials_are_masked():
    assert camera._mask_source("rtsp://u:p@h:554/s1") == "rtsp://***:***@h:554/s1"
    assert camera._mask_source("rtsp://h:554/s1") == "rtsp://h:554/s1"
    assert camera._mask_source(0) == "0"  # USB の番号はそのまま


def test_the_capture_thread_logs_through_the_mask():
    """`source` をログへ渡す行は、すべて伏せる関数を通る。"""
    src = inspect.getsource(camera)
    bare = [
        line.strip()
        for line in src.splitlines()
        if "logger." in line and ", source)" in line and "_mask_source(" not in line
    ]
    assert not bare, f"認証情報を伏せずにログへ出している: {bare}"
