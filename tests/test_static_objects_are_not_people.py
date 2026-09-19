"""定点の静止物を人と数えない（知-v・2026-09-18・`知覚在席` v0.25）。

出入口の定点で YOLO が右下の暗い塊（家具か布）を 7 分間「1 人」と読み続け、在席表も `/speaker` の
指定も切れなかった（実機 21:50〜21:57・写真は同じ絵・動体イベントも無し）。閾値を上げるだけだと本当の
人も落ちる。**人は動く**：前回の枠と重なり（IoU）0.9〔仮〕以上で対応づく枠は「動かない時間」を
引き継ぎ、300 秒〔仮〕以上動かない枠は数えない。動けば 0 から。動体イベントでも 0 から。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.core import presence_rules
from familiar_agent.core.presence_rules import StaticBoxes, count_moving, iou

BOX = (100.0, 200.0, 180.0, 400.0)  # 静止物
MAN = (500.0, 100.0, 620.0, 420.0)


def test_iou_of_the_same_box_is_one_and_disjoint_is_zero():
    assert iou(BOX, BOX) == 1.0
    assert iou(BOX, MAN) == 0.0
    assert 0.8 < iou(BOX, (102.0, 202.0, 182.0, 402.0)) < 1.0


def test_a_box_that_never_moves_stops_counting_after_static_sec():
    st = StaticBoxes()
    n, st = count_moving(st, [BOX], now=0.0, static_sec=300.0, min_iou=0.9)
    assert n == 1  # 最初は人として数える（まだ分からない）
    n, st = count_moving(st, [BOX], now=299.0, static_sec=300.0, min_iou=0.9)
    assert n == 1
    n, st = count_moving(st, [BOX], now=300.0, static_sec=300.0, min_iou=0.9)
    assert n == 0  # 300 秒動かない → 物
    n, st = count_moving(st, [BOX], now=900.0, static_sec=300.0, min_iou=0.9)
    assert n == 0


def test_a_box_that_moves_counts_again_from_zero():
    st = StaticBoxes()
    _, st = count_moving(st, [BOX], now=0.0, static_sec=300.0, min_iou=0.9)
    _, st = count_moving(st, [BOX], now=400.0, static_sec=300.0, min_iou=0.9)
    moved = (160.0, 200.0, 240.0, 400.0)  # 右へ 60 px（IoU < 0.9）
    n, st = count_moving(st, [moved], now=430.0, static_sec=300.0, min_iou=0.9)
    assert n == 1
    n, st = count_moving(st, [moved], now=700.0, static_sec=300.0, min_iou=0.9)
    assert n == 1  # 430 から 270 秒・まだ人


def test_a_real_person_beside_a_static_object_is_counted():
    st = StaticBoxes()
    _, st = count_moving(st, [BOX], now=0.0, static_sec=300.0, min_iou=0.9)
    _, st = count_moving(st, [BOX], now=400.0, static_sec=300.0, min_iou=0.9)
    n, st = count_moving(st, [BOX, MAN], now=430.0, static_sec=300.0, min_iou=0.9)
    assert n == 1  # 物 1・人 1
    n, st = count_moving(st, [BOX], now=460.0, static_sec=300.0, min_iou=0.9)
    assert n == 0  # 人が去った


def test_a_motion_event_resets_the_static_clock():
    st = StaticBoxes()
    _, st = count_moving(st, [BOX], now=0.0, static_sec=300.0, min_iou=0.9)
    _, st = count_moving(st, [BOX], now=400.0, static_sec=300.0, min_iou=0.9)
    st = presence_rules.reset(st)
    n, st = count_moving(st, [BOX], now=430.0, static_sec=300.0, min_iou=0.9)
    assert n == 1


# ── センサ：枠で受け、定点ごとに状態を持つ ────────────────────────────────────


def test_the_sensor_uses_boxes_and_stops_counting_a_static_object(monkeypatch):
    from familiar_agent.poses import Pose
    from familiar_agent.presence_sensor import PresenceSensor

    camera = MagicMock()
    camera.position = AsyncMock(return_value=(0.0, -0.5))
    camera.capture = AsyncMock(return_value=("B64", "/tmp/f.jpg"))
    detector = MagicMock()
    detector.boxes = AsyncMock(return_value=[BOX])
    s = PresenceSensor(
        camera=camera,
        poses_getter=AsyncMock(return_value=[Pose("正面", 0.0, -0.5)]),
        detector=detector,
        tolerance=0.02,
        window_sec=120.0,
        interval_sec=30.0,
        static_sec=300.0,
        static_iou=0.9,
    )
    clock = {"t": 1000.0}
    monkeypatch.setattr("familiar_agent.presence_sensor.time.time", lambda: clock["t"])
    asyncio.run(s.check_once())
    assert s._map._seen["正面"] == 1000.0
    clock["t"] = 1200.0
    asyncio.run(s.check_once())
    assert s._map._seen["正面"] == 1200.0  # 200 秒・まだ人
    clock["t"] = 1310.0
    asyncio.run(s.check_once())
    assert s._map._seen["正面"] == 1200.0  # 310 秒動かない → 数えない（印を付けない）
    detector.count.assert_not_called()  # 数でなく枠で受ける
    s.on_motion()  # 動体 → 積算を捨てる
    clock["t"] = 1340.0
    asyncio.run(s.check_once())
    assert s._map._seen["正面"] == 1340.0


def test_the_presence_window_is_sixty_seconds(monkeypatch):
    """知-x（2026-09-19）：1 フレームの誤検出が滞留窓（180 秒）を延ばし、無人でも「居る」が 3 分続いた。
    窓を 60 秒に。静止している人が 60 秒検出されない見落としは受ける（本人の決定・反応の鈍さを嫌う）。"""
    from familiar_agent.config import CameraConfig

    monkeypatch.delenv("CAMERA_PRESENCE_WINDOW", raising=False)
    assert CameraConfig().presence_window_sec == 60.0
