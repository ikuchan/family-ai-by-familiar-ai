"""カメラ側の動体イベント（ONVIF PullPoint）。

こちらから撮りに行くのではなく、**カメラが「動いた」と言ってきたとき**に起こす。実機
（Tapo C211・ファーム 1.2.6）で確認したトピックは `CellMotionDetector/Motion`（`IsMotion`）と
`TamperDetector/Tamper` の2つだけで、**人かどうかは分からない**（仕様上は人検出を持つが
ONVIF には出ていない）。人の判定は YOLO が担う。

実機の癖：`create_pullpoint_service()` はそのままでは失敗する（カメラが能力一覧に宛先を
載せない）ので、購読で返る宛先を `xaddrs` へ入れてから作る。接続はよく切れるので、切断は
異常ではなく通常の流れとして扱い、間隔を空けて購読し直す。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.recognition.motion_events import MotionEventWatcher, _backoff, _count_motion

_MOTION = "tns1:RuleEngine/CellMotionDetector/Motion"
_TAMPER = "tns1:RuleEngine/TamperDetector/Tamper"


def _note(topic, name="IsMotion", value="true"):
    n = MagicMock()
    n.Topic._value_1 = topic
    item = MagicMock()
    item.Name, item.Value = name, value
    n.Message._value_1.Data.SimpleItem = [item]
    return n


# --- 通知の読み取り -------------------------------------------------------


def test_a_motion_notification_counts():
    assert _count_motion([_note(_MOTION)]) == 1


def test_motion_stopping_is_ignored():
    """案A：`IsMotion=false` は使わない。

    静止している人が居るので「動きが止まった＝居なくなった」ではない。不在は滞留窓と
    YOLO が決める。
    """
    assert _count_motion([_note(_MOTION, value="false")]) == 0


def test_tampering_is_not_motion():
    assert _count_motion([_note(_TAMPER)]) == 0


def test_a_notification_without_the_expected_shape_is_skipped():
    # 機種やファームで形が変わる。読めないものでループを殺さない。
    assert _count_motion([MagicMock(spec=[])]) == 0


def test_nothing_pulled_means_no_motion():
    assert _count_motion([]) == 0
    assert _count_motion(None) == 0


# --- 再試行の間隔 ---------------------------------------------------------


def test_the_wait_doubles_after_each_failure():
    assert [_backoff(i) for i in range(5)] == [1.0, 2.0, 4.0, 8.0, 16.0]


def test_the_wait_stops_growing_at_the_cap():
    # 案あ：諦めずに試し続ける。上限 128 秒（カメラの再起動から自動で戻れる間隔）。
    assert _backoff(7) == 128.0
    assert _backoff(20) == 128.0


# --- 購読 -----------------------------------------------------------------


def _onvif(pull_returns):
    """購読して PullMessages を返す ONVIF の作り。"""
    onvif = MagicMock()
    onvif.xaddrs = {}
    ev = MagicMock()
    sub = MagicMock()
    sub.SubscriptionReference.Address._value_1 = "http://192.0.2.1:1024/event-1"
    ev.CreatePullPointSubscription = AsyncMock(return_value=sub)
    onvif.create_events_service = AsyncMock(return_value=ev)
    pull = MagicMock()
    pull.PullMessages = AsyncMock(side_effect=pull_returns)
    onvif.create_pullpoint_service = AsyncMock(return_value=pull)
    return onvif, pull


def _messages(notes):
    m = MagicMock()
    m.NotificationMessage = notes
    return m


def test_the_subscription_address_is_injected_before_making_the_service():
    """実機の癖：カメラが能力一覧に PullPoint の宛先を載せないので、手で入れる。"""
    onvif, _ = _onvif([_messages([])])
    w = MotionEventWatcher(lambda: onvif, on_motion=lambda: None)
    asyncio.run(w._subscribe())
    assert "http://192.0.2.1:1024/event-1" in onvif.xaddrs.values()


def test_motion_reaches_the_callback():
    fired = []
    onvif, _ = _onvif([_messages([_note(_MOTION)]), asyncio.CancelledError()])
    w = MotionEventWatcher(lambda: onvif, on_motion=lambda: fired.append(1))

    async def go():
        await w.start()
        await asyncio.sleep(0.05)
        await w.stop()

    asyncio.run(go())
    assert fired == [1]


def test_a_disconnect_leads_to_a_new_subscription_rather_than_a_dead_task():
    # 実機は3回に2回切れた。切断で常駐タスクが死ぬと、以後ずっと気づけない。
    onvif, _ = _onvif([
        RuntimeError("Server disconnected"),
        _messages([_note(_MOTION)]),
        asyncio.CancelledError(),
    ])
    fired = []
    w = MotionEventWatcher(lambda: onvif, on_motion=lambda: fired.append(1))
    w._wait_for = AsyncMock()          # 再試行の待ちを飛ばす

    async def go():
        await w.start()
        await asyncio.sleep(0.05)
        await w.stop()

    asyncio.run(go())
    assert fired == [1]
    assert onvif.create_events_service.await_count >= 2


def test_stopping_leaves_no_task_behind():
    onvif, _ = _onvif([_messages([])])
    w = MotionEventWatcher(lambda: onvif, on_motion=lambda: None)

    async def go():
        await w.start()
        await w.stop()
        return w._task

    assert asyncio.run(go()) is None


def test_without_a_camera_the_watcher_simply_does_not_run():
    w = MotionEventWatcher(lambda: None, on_motion=lambda: None)

    async def go():
        await w.start()
        await asyncio.sleep(0.02)
        await w.stop()

    asyncio.run(go())   # 例外を出さないこと


def test_the_onvif_source_may_need_connecting_first():
    """`_cam_onvif` は `_ensure_connected()` を呼ぶまで `None` である。

    起動直後は誰も呼んでいないので、素の属性を渡すと購読は「カメラが無い構成」とみなして
    静かに終わる（実機で観測：購読のログが一度も出なかった）。繋いでから渡せるよう、
    待てる値も受け取る。
    """
    onvif, _ = _onvif([_messages([])])

    async def provider():
        return onvif

    w = MotionEventWatcher(provider, on_motion=lambda: None)
    asyncio.run(w._subscribe())
    assert onvif.create_events_service.await_count == 1
