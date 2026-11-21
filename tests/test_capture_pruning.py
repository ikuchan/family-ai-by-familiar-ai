"""環-bc：カメラの画像を 7 日で消す（本人の決定、2026-11-21）。

在席の見回り（3 秒おき）が撮るたびに `CameraTool.capture` が画像を書き、消す仕組みが無かったので、8/30 から約 108 万枚、
125 GB 溜まった。画像を使うのは撮ったときの 1 回の求めのあいだだけで、記録には縮めた画像が DB に入る。
古さはファイルの名前（撮ったときの時刻）で比べる。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import threading
import time

from familiar_agent.config import CameraConfig
from familiar_agent.tools import camera as cam_mod

#: 共通の守り（`conftest.py`）が消す仕事を何もしない口に替える前の、本物の口。
_REAL_START_PRUNE = cam_mod.CameraTool._start_prune

_NOW = datetime(2026, 11, 21, 12, 0, 0)


def _name(age: timedelta) -> str:
    return f"capture_{(_NOW - age):%Y%m%d_%H%M%S}.jpg"


def test_old_captures_are_deleted_and_new_ones_and_other_files_are_kept(tmp_path) -> None:
    (tmp_path / _name(timedelta(days=8))).write_bytes(b"x" * 10)
    (tmp_path / _name(timedelta(days=6))).write_bytes(b"x" * 10)
    (tmp_path / "notes.txt").write_bytes(b"x")  # 画像でないものには触らない
    (tmp_path / "capture_読めない名前.png").write_bytes(b"x")
    (tmp_path / "capture_0.jpg").write_bytes(b"x")  # 時刻の形でない名前は、古く見えても消さない
    n = cam_mod.prune_captures(tmp_path, keep_sec=7 * 86_400.0, now=_NOW)
    assert n == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [_name(timedelta(days=6)), "notes.txt", "capture_読めない名前.png", "capture_0.jpg"]
    )


def test_deleting_many_pauses_every_thousand(tmp_path) -> None:
    for i in range(2500):
        (tmp_path / _name(timedelta(days=10, seconds=i))).write_bytes(b"")
    pauses: list[int] = []
    n = cam_mod.prune_captures(
        tmp_path, keep_sec=86_400.0, now=_NOW, pause=lambda: pauses.append(1)
    )
    assert n == 2500 and len(pauses) == 2


def test_a_missing_folder_is_fine(tmp_path) -> None:
    assert cam_mod.prune_captures(tmp_path / "無い", keep_sec=1.0, now=_NOW) == 0


def test_a_raised_stop_ends_the_sweep_early(tmp_path) -> None:
    for i in range(10):
        (tmp_path / _name(timedelta(days=10, seconds=i))).write_bytes(b"")
    stop = threading.Event()
    stop.set()
    assert cam_mod.prune_captures(tmp_path, keep_sec=86_400.0, now=_NOW, stop=stop) == 0


def test_the_defaults_keep_seven_days_and_prune_hourly() -> None:
    assert CameraConfig.capture_keep_days == 7.0
    assert CameraConfig.capture_prune_interval_sec == 3600.0


def test_making_a_camera_starts_the_pruning_with_the_given_values(monkeypatch) -> None:
    """カメラを作ると消す仕事が 1 本立つ。残す日数は設定から渡す（共通の守りは外して確かめる）。"""
    started: list[float] = []
    monkeypatch.setattr(
        cam_mod.CameraTool, "_start_prune", lambda self: started.append(self._keep_sec)
    )
    cam_mod.CameraTool("192.0.2.1", "u", "p", 2020, capture_keep_days=2.0)
    assert started == [2 * 86_400.0]


def test_the_pruning_waits_first_and_stops_when_the_camera_closes(monkeypatch) -> None:
    """本物の消す仕事：最初も間隔のあとに消し、閉じると止まる。"""
    calls: list[float] = []
    monkeypatch.setattr(
        cam_mod, "prune_captures", lambda *a, **k: calls.append(time.monotonic()) or 0
    )
    cam = cam_mod.CameraTool("192.0.2.1", "u", "p", 2020, capture_prune_interval_sec=0.05)
    begun = time.monotonic()
    _REAL_START_PRUNE(cam)
    time.sleep(0.3)
    cam.close()
    n = len(calls)
    time.sleep(0.2)
    assert n >= 2 and calls[0] - begun >= 0.04  # 最初の 1 回も間隔のあと
    assert len(calls) <= n + 1  # 閉じたあとは回らない（閉じる瞬間に回っていた 1 回まで）


def test_a_zero_interval_does_not_prune(monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(cam_mod, "prune_captures", lambda *a, **k: calls.append(1) or 0)
    cam = cam_mod.CameraTool("192.0.2.1", "u", "p", 2020, capture_prune_interval_sec=0)
    _REAL_START_PRUNE(cam)
    time.sleep(0.1)
    cam.close()
    assert calls == []
