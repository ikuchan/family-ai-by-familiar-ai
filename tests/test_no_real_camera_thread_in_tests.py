"""テストの中では、カメラの本物の裏方スレッドを立てない（2026-09-25）。

`CameraTool` は作った瞬間に裏方スレッド（`_capture_loop`）を立て、映像の接続を開きに行く。
`test_camera_absolute.py`・`test_look_at_pose.py` は届かないアドレス（192.0.2.1）で作るので、
テスト 1 件ごとに 1 本、計 17 本のスレッドが 30 秒待って時間切れになる。**その瞬間がプロセスの
終了と重なると、全件通過のあとに終了コード 134（`terminate called without an active exception`）で
落ちた**（2026-09-25 の全体テストで 3 回・終了を 23.9 秒遅らせて再現）。

守りは全テスト共通（`conftest.py` の `_no_real_camera_thread`）。
"""

from __future__ import annotations

from familiar_agent.tools.camera import CameraTool


def test_making_a_camera_starts_no_capture_thread():
    cam = CameraTool("192.0.2.1", "u", "p", 2020)
    assert cam._thread is None
    assert cam._running is False
