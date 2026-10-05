"""定点の静止物を人と数えない（知-v・2026-09-18・`知覚在席` v0.25）。

出入口の定点で YOLO が右下の暗い塊（家具か布）を 7 分間「1 人」と読み続け、顔ぶれ表も `/speaker` の
指定も切れなかった（実機 21:50〜21:57・写真は同じ絵・動体イベントも無し）。閾値を上げるだけだと本当の
人も落ちる。**人は動く**：前回の枠と重なり（IoU）0.9〔仮〕以上で対応づく枠は「動かない時間」を
引き継ぎ、300 秒〔仮〕以上動かない枠は数えない。動けば 0 から。動体イベントでも 0 から。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.core import occupancy_rules
from familiar_agent.core.occupancy_rules import StaticBoxes, count_moving, iou

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
    st = occupancy_rules.reset(st)
    n, st = count_moving(st, [BOX], now=430.0, static_sec=300.0, min_iou=0.9)
    assert n == 1


# ── センサ：枠で受け、定点ごとに状態を持つ ────────────────────────────────────


def test_the_sensor_uses_boxes_and_stops_counting_a_static_object(monkeypatch):
    from familiar_agent.poses import Pose
    from familiar_agent.occupancy_sensor import OccupancySensor

    camera = MagicMock()
    camera.position = AsyncMock(return_value=(0.0, -0.5))
    camera.capture = AsyncMock(return_value=("B64", "/tmp/f.jpg"))
    detector = MagicMock()
    detector.boxes = AsyncMock(return_value=[BOX])
    s = OccupancySensor(
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
    monkeypatch.setattr("familiar_agent.occupancy_sensor.time.time", lambda: clock["t"])
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


# ── 枠が抜けても「動かない時間」を持ち越す（知-v-ろ・2026-10-06）──────────────
#
# 実機 2026-10-06 02:41〜06:32、テレビの定点のレンズ前の物を YOLO が確かさ 0.25〜0.34 で「人」と読んだ。
# 枠は動かない（重なり 0.987 以上）が、読めるのは 2 枚に 1 枚ほどで、1 回抜けるたびに積算が空に戻り、
# 「1 人」が 4 時間続いた（途切れずに続いたのは最長 25 秒）。抜けは最長 80 秒だったので、最後に見えてから
# 90 秒〔仮〕までは前の枠と起点を持ち越す。持ち越した枠は数えない。

G = dict(static_sec=300.0, min_iou=0.9, grace_sec=90.0)


def test_a_gap_keeps_the_static_clock():
    st = StaticBoxes()
    _, st = count_moving(st, [BOX], now=0.0, **G)
    _, st = count_moving(st, [BOX], now=3.0, **G)
    for t in range(6, 60, 3):
        _, st = count_moving(st, [], now=float(t), **G)
    n, st = count_moving(st, [BOX], now=60.0, **G)
    assert n == 1  # まだ 60 秒
    n, st = count_moving(st, [BOX], now=300.0, **G)
    assert n == 0  # 起点 0 を引き継いで 300 秒 → 物


def test_a_gap_longer_than_the_grace_starts_from_zero():
    st = StaticBoxes()
    _, st = count_moving(st, [BOX], now=0.0, **G)
    _, st = count_moving(st, [BOX], now=200.0, **G)
    _, st = count_moving(st, [], now=250.0, **G)
    _, st = count_moving(st, [], now=291.0, **G)  # 最後に見えてから 91 秒
    n, st = count_moving(st, [BOX], now=300.0, **G)
    assert n == 1  # 0 から
    n, st = count_moving(st, [BOX], now=599.0, **G)
    assert n == 1


def test_a_carried_box_is_not_counted():
    st = StaticBoxes()
    _, st = count_moving(st, [MAN], now=0.0, **G)
    n, st = count_moving(st, [], now=3.0, **G)
    assert n == 0  # 見えていない枠は人として数えない


def test_tonights_flicker_stops_counting_after_static_sec():
    """昨夜の形：読めたり抜けたり、抜けは最長 80 秒。300 秒をすぎたら最後まで数えない。"""
    gaps = [3, 6, 3, 10, 20, 3, 30, 45, 3, 80, 6, 9, 60, 3, 12] * 8
    st = StaticBoxes()
    t = 0.0
    counted_after = 0
    for g in gaps:
        n, st = count_moving(st, [BOX], now=t, **G)
        if t >= 300.0:
            counted_after += n
        for k in range(3, g, 3):
            _, st = count_moving(st, [], now=t + k, **G)
        t += g
    assert t > 2000.0 and counted_after == 0


def test_a_motion_event_drops_carried_boxes_too():
    st = StaticBoxes()
    _, st = count_moving(st, [BOX], now=0.0, **G)
    _, st = count_moving(st, [], now=280.0, **G)
    st = occupancy_rules.reset(st)
    n, st = count_moving(st, [BOX], now=300.0, **G)
    assert n == 1


def test_the_grace_is_set_and_reaches_the_rules(monkeypatch):
    from familiar_agent.config import CameraConfig
    from familiar_agent.occupancy_sensor import OccupancySensor
    from familiar_agent.poses import Pose

    monkeypatch.delenv("OCCUPANCY_STATIC_GRACE_SEC", raising=False)
    assert CameraConfig().occupancy_static_grace_sec == 90.0
    camera = MagicMock()
    camera.position = AsyncMock(return_value=(0.0, -0.5))
    camera.capture = AsyncMock(return_value=("B64", "/tmp/f.jpg"))
    detector = MagicMock()
    s = OccupancySensor(
        camera=camera,
        poses_getter=AsyncMock(return_value=[Pose("正面", 0.0, -0.5)]),
        detector=detector,
        tolerance=0.02,
        window_sec=30.0,
        interval_sec=3.0,
        static_sec=300.0,
        static_iou=0.9,
        static_grace_sec=90.0,
    )
    clock = {"t": 1000.0}
    monkeypatch.setattr("familiar_agent.occupancy_sensor.time.time", lambda: clock["t"])
    for t, boxes in [(1000.0, [BOX]), (1060.0, []), (1120.0, [BOX]), (1300.0, [BOX])]:
        clock["t"] = t
        detector.boxes = AsyncMock(return_value=boxes)
        asyncio.run(s.check_once())
    assert s.last_reading()[1] == 0  # 抜けを挟んで 300 秒 → 物


def test_the_agent_passes_the_grace():
    import inspect

    from familiar_agent import agent

    assert "static_grace_sec=cam_cfg.occupancy_static_grace_sec" in inspect.getsource(agent)
