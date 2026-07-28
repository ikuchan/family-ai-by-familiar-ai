"""在/不在を実際に測る器（撮る → 定点を特定 → YOLO → マップ更新）。

`知覚在席` §3-2 の在/不在は **G（T 側・連続）** が担う。ここが部品を束ねる場所で、
S3a（`presence_map`・`person_detector`）と S2（定点）と S3b（動体イベント）を繋ぐ。

起こされ方は2つある。**カメラが「動いた」と言ってきたとき**と、**一定間隔（30秒）**。
動体だけでは足りない。静止している人は動体を出さないので、動きが無いあいだも確かめる。

**どの定点を見ているか分からないときは、何も記録しない。** 向きが読めない、または定点から
離れている（移動中）ときに記録すると、別の場所の映像でその定点の在席が汚れる
（`知覚在席` §3-3 の振動中ゲート）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.poses import Pose
from familiar_agent.presence_sensor import PresenceSensor

_POSES = [Pose("窓側", 0.0, -0.5), Pose("出入り口", -0.129, -0.5)]


def _sensor(*, position=(-0.129, -0.5), people=0, capture=("BASE64", "/tmp/f.jpg"),
            poses=None):
    camera = MagicMock()
    camera.position = AsyncMock(return_value=position)
    camera.capture = AsyncMock(return_value=capture)
    detector = MagicMock()
    detector.count = AsyncMock(return_value=people)
    s = PresenceSensor(
        camera=camera,
        poses_getter=AsyncMock(return_value=_POSES if poses is None else poses),
        detector=detector,
        tolerance=0.02,
        window_sec=120.0,
        interval_sec=30.0,
    )
    return s, camera, detector


# --- 1回の確認 -------------------------------------------------------------


def test_seeing_a_person_marks_that_pose_as_occupied():
    s, _, _ = _sensor(people=1)
    assert asyncio.run(s.check_once()) == "出入り口"
    assert s.room_occupied() is True
    assert s.poses_seen() == ["出入り口"]


def test_seeing_nobody_leaves_the_room_empty():
    s, _, _ = _sensor(people=0)
    assert asyncio.run(s.check_once()) == "出入り口"
    assert s.room_occupied() is False


def test_the_frame_is_only_analysed_once_per_check():
    s, camera, detector = _sensor(people=1)
    asyncio.run(s.check_once())
    assert camera.capture.await_count == 1
    assert detector.count.await_count == 1


# --- 振動中ゲート ---------------------------------------------------------


def test_a_position_between_poses_records_nothing():
    # 移動中の映像は、どの定点のものでもない。
    s, _, detector = _sensor(position=(-0.30, -0.5), people=1)
    assert asyncio.run(s.check_once()) is None
    assert s.room_occupied() is False
    detector.count.assert_not_awaited()


def test_an_unreadable_position_records_nothing():
    # 向きが読めないのに記録すると、別の場所の映像で在席が汚れる。
    s, _, detector = _sensor(position=None, people=1)
    assert asyncio.run(s.check_once()) is None
    detector.count.assert_not_awaited()


def test_without_any_poses_nothing_is_recorded():
    s, _, _ = _sensor(people=1, poses=[])
    assert asyncio.run(s.check_once()) is None


# --- 壊れたときの振る舞い -------------------------------------------------


def test_a_failed_capture_records_nothing_rather_than_absence():
    # 撮れなかったことを「誰も居ない」にすると、カメラの不調で人が消える。
    s, camera, detector = _sensor(people=1, capture=(None, None))
    assert asyncio.run(s.check_once()) is None
    detector.count.assert_not_awaited()


def test_a_camera_that_raises_does_not_kill_the_caller():
    s, camera, _ = _sensor()
    camera.position = AsyncMock(side_effect=RuntimeError("ptz offline"))
    assert asyncio.run(s.check_once()) is None


# --- 見た映像の受け渡し ---------------------------------------------------


def test_the_latest_frame_is_kept_for_the_gui():
    """GUI は在席確認のカメラ映像として直近フレームを表示する。

    顔ベースの常駐（`presence_watcher`）を止めるので、その供給元をここが引き継ぐ。
    止めたまま供給しないと、GUI から映像が消える。
    """
    s, _, _ = _sensor(people=0)
    asyncio.run(s.check_once())
    assert s.latest_frame_b64() == "BASE64"


# --- 起こされ方 -----------------------------------------------------------


def test_motion_triggers_a_check_without_waiting_for_the_interval():
    s, camera, _ = _sensor(people=1)

    async def go():
        await s.start()
        s.on_motion()                 # カメラが「動いた」と言ってきた
        await asyncio.sleep(0.05)
        await s.stop()

    asyncio.run(go())
    assert camera.capture.await_count >= 1


def test_stopping_leaves_no_task_behind():
    s, _, _ = _sensor()

    async def go():
        await s.start()
        await s.stop()
        return s._task

    assert asyncio.run(go()) is None


# --- 確認の下限間隔 -------------------------------------------------------


def test_the_first_event_fires_at_once():
    """動き始めには即座に気づきたい。前回の確認から間が空いていれば待たない。"""
    from familiar_agent.presence_sensor import _delay_before

    assert _delay_before(last_check=0.0, now=10.0, min_gap=3.0) == 0.0


def test_events_that_keep_coming_are_thinned_out():
    """動いているあいだイベントは毎秒何件も飛ぶ。

    そのたびに撮って YOLO を回すと、実機では 0.15 秒ごとに確認していた（電力と発熱の面で
    放置できない）。前回の確認からの残り時間だけ待つ。
    """
    from familiar_agent.presence_sensor import _delay_before

    assert abs(_delay_before(last_check=10.0, now=10.2, min_gap=3.0) - 2.8) < 1e-9


def test_the_wait_never_goes_negative():
    from familiar_agent.presence_sensor import _delay_before

    assert _delay_before(last_check=0.0, now=100.0, min_gap=3.0) == 0.0


# --- 見えの「普通」 -------------------------------------------------------


def _sensor_with_norm(*, people=0, norm=None, observations=0, embedding=None):
    s, camera, detector = _sensor(people=people)
    encoder = MagicMock()
    encoder.embed = AsyncMock(return_value=embedding if embedding is not None else [0.5] * 384)
    store = MagicMock()
    store.load = MagicMock(return_value=(norm, observations))
    store.save = MagicMock()
    s.attach_visual_norm(encoder, store)
    return s, encoder, store


def test_the_norm_grows_on_every_check():
    s, _, store = _sensor_with_norm(norm=[0.5] * 384, observations=3)
    asyncio.run(s.check_once())
    assert store.save.call_args.kwargs["observations"] == 4


def test_no_surprise_is_reported_before_the_norm_has_grown():
    """5回見るまでは比較対象として使わない。

    最初の1枚をそのまま「普通」にすると、そのとき写っていたものが基準になる。
    """
    s, _, _ = _sensor_with_norm(norm=[0.5] * 384, observations=1)
    asyncio.run(s.check_once())
    assert s.scene_surprise() is None


def test_a_grown_norm_yields_a_distance():
    s, _, _ = _sensor_with_norm(norm=[1.0] + [0.0] * 383, observations=5,
                                embedding=[0.0, 1.0] + [0.0] * 382)
    asyncio.run(s.check_once())
    assert s.scene_surprise() is not None and s.scene_surprise() > 0.9


def test_an_encoder_that_fails_does_not_stop_the_presence_check():
    # 見えが取れなくても、人が居るかどうかは分かる。
    s, encoder, store = _sensor_with_norm(people=1, norm=[0.5] * 384, observations=5)
    encoder.embed = AsyncMock(return_value=None)
    assert asyncio.run(s.check_once()) == "出入り口"
    assert s.room_occupied() is True
    store.save.assert_not_called()


def test_without_an_encoder_the_sensor_works_as_before():
    s, _, _ = _sensor(people=1)
    assert asyncio.run(s.check_once()) == "出入り口"
    assert s.scene_surprise() is None
