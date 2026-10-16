"""在席が「居る」から「居ない」に変わったら、そのあいだの読みを 1 行残す（知-bg 段 1・2026-10-16・本人の決定ア）。

10/07〜10/14 に名前で呼んだ声の 4 件で「居ない」になった。人が消えた理由（枠なし・動かないので物とみなした・撮れ
なかった・別の向き）は DEBUG にしか出ず、分けられなかった。切り替わりのときだけ INFO に 1 行残し、また居るように
なったら、居なかった長さを 1 行残す。判定そのものは変えない。
"""

from __future__ import annotations

import asyncio
import time as real_time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent import occupancy_sensor as mod
from familiar_agent.occupancy_sensor import OccupancySensor
from familiar_agent.poses import Pose

_POSES = [Pose("窓側", 0.0, -0.5), Pose("出入り口", -0.129, -0.5)]
_AT_DOOR = (-0.129, -0.5)
_BOX = [(0.0, 0.0, 100.0, 300.0)]


@pytest.fixture
def clock(monkeypatch):
    now = [1_000_000.0]
    monkeypatch.setattr(
        mod, "time", SimpleNamespace(time=lambda: now[0], monotonic=real_time.monotonic)
    )
    return now


def _sensor(*, static_sec=300.0):
    camera = MagicMock()
    camera.position = AsyncMock(return_value=_AT_DOOR)
    camera.capture = AsyncMock(return_value=("BASE64", "/tmp/f.jpg"))
    detector = MagicMock()
    detector.boxes = AsyncMock(return_value=[])
    s = OccupancySensor(
        camera=camera,
        poses_getter=AsyncMock(return_value=_POSES),
        detector=detector,
        tolerance=0.02,
        window_sec=30.0,
        interval_sec=3.0,
        static_sec=static_sec,
    )
    return s, camera, detector


def _see(s, detector, boxes):
    detector.boxes.return_value = boxes
    asyncio.run(s.check_once())


def _lines(caplog, head):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(head)]


def test_the_moment_it_empties_says_what_was_read(clock, caplog):
    s, _, detector = _sensor()
    with caplog.at_level("INFO", logger="familiar_agent.occupancy_sensor"):
        _see(s, detector, _BOX)
        for _ in range(11):  # 3 秒ごとに枠なし。33 秒で窓（30 秒）の外
            clock[0] += 3
            _see(s, detector, [])
    got = _lines(caplog, "在席：居なくなった")
    assert len(got) == 1
    assert "出入り口" in got[0] and "33 秒前" in got[0]
    assert "枠なし 11" in got[0] and "動かないので数えない 0" in got[0]


def test_a_still_person_shows_as_not_counted(clock, caplog):
    s, _, detector = _sensor(static_sec=5.0)
    with caplog.at_level("INFO", logger="familiar_agent.occupancy_sensor"):
        for _ in range(13):  # 同じ枠が動かない。5 秒から先は物とみなす
            _see(s, detector, _BOX)
            clock[0] += 3
        _see(s, detector, _BOX)
    got = _lines(caplog, "在席：居なくなった")
    assert len(got) == 1
    assert "動かないので数えない" in got[0] and "動かないので数えない 0" not in got[0]


def test_each_kind_of_reading_is_counted(clock, caplog):
    s, camera, detector = _sensor()
    with caplog.at_level("INFO", logger="familiar_agent.occupancy_sensor"):
        _see(s, detector, _BOX)
        steps = [
            ("pos", None),
            ("pos", (-0.30, -0.5)),
            ("cap", ("", "")),
            ("cap", ("", "")),
            ("boom", None),
        ]
        for kind, value in steps:
            clock[0] += 7
            camera.position.return_value = _AT_DOOR
            camera.capture.return_value = ("BASE64", "/tmp/f.jpg")
            camera.capture.side_effect = None
            if kind == "pos":
                camera.position.return_value = value
            elif kind == "cap":
                camera.capture.return_value = value
            else:
                camera.capture.side_effect = RuntimeError("壊れた")
            asyncio.run(s.check_once())
    got = _lines(caplog, "在席：居なくなった")
    assert len(got) == 1
    for part in ("向きが読めない 1", "移動中 1", "撮れなかった 2", "確認に失敗 1", "枠なし 0"):
        assert part in got[0]


def test_coming_back_says_how_long_it_was_empty(clock, caplog):
    s, _, detector = _sensor()
    with caplog.at_level("INFO", logger="familiar_agent.occupancy_sensor"):
        _see(s, detector, _BOX)
        clock[0] += 40
        _see(s, detector, [])  # ここで居なくなった
        clock[0] += 92
        _see(s, detector, _BOX)
    back = _lines(caplog, "在席：また居る")
    assert back == ["在席：また居る（出入り口・居なかったのは 132 秒）"]


def test_the_first_sighting_has_no_length(clock, caplog):
    s, _, detector = _sensor()
    with caplog.at_level("INFO", logger="familiar_agent.occupancy_sensor"):
        _see(s, detector, _BOX)
    assert _lines(caplog, "在席：また居る") == []
    assert _lines(caplog, "在席：居なくなった") == []


def test_inside_the_window_nothing_is_said(clock, caplog):
    """**反証の側**：30 秒の内側で読みが空になっても、まだ居る。"""
    s, _, detector = _sensor()
    with caplog.at_level("INFO", logger="familiar_agent.occupancy_sensor"):
        _see(s, detector, _BOX)
        for _ in range(9):
            clock[0] += 3
            _see(s, detector, [])
    assert s.room_occupied() is True
    assert _lines(caplog, "在席：居なくなった") == []


def test_staying_empty_says_it_only_once(clock, caplog):
    """**反証の側**：居ないまま確認が続いても、切り替わりの 1 回だけ。"""
    s, _, detector = _sensor()
    with caplog.at_level("INFO", logger="familiar_agent.occupancy_sensor"):
        _see(s, detector, _BOX)
        for _ in range(30):
            clock[0] += 3
            _see(s, detector, [])
    assert len(_lines(caplog, "在席：居なくなった")) == 1
