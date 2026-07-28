"""カメラ側の動体イベントを購読する（ONVIF PullPoint）。

こちらから撮りに行くのではなく、**カメラが「動いた」と言ってきたとき**に起こす。実機
（Tapo C211・ファーム 1.2.6）が持つトピックは `CellMotionDetector/Motion`（`IsMotion`）と
`TamperDetector/Tamper` の2つで、**人かどうかは分からない**。仕様上はカメラも人検出を
持つが ONVIF には出てこないので、人の判定は YOLO（`person_detector`）が担う。

実機で確かめた癖が2つある。

- `create_pullpoint_service()` はそのままでは失敗する。カメラが能力一覧に PullPoint の
  宛先を載せないので、**購読で返ってきた宛先を `xaddrs` へ入れてから**サービスを作る。
- **接続がよく切れる**（3回中2回が `ServerDisconnectedError`）。切断は異常ではなく通常の
  流れとして扱い、間隔を空けて購読し直す。諦めない（カメラの再起動からも自動で戻る）。

`IsMotion=false`（動きが止まった）は使わない。静止している人が居るので「動きが止まった＝
居なくなった」ではない。不在は滞留窓と YOLO が決める。
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_MOTION_TOPIC = "CellMotionDetector/Motion"
_PULLPOINT_NS = "http://www.onvif.org/ver10/events/wsdl/PullPointSubscription"
# 購読の有効期限。切れる前に取り直す。
_TERMINATION = "PT120S"
_PULL_TIMEOUT = "PT10S"
_PULL_LIMIT = 10
# 再試行の待ちは倍々に伸ばし、この秒数で頭打ちにする。
_BACKOFF_CAP = 128.0


def _backoff(attempt: int) -> float:
    """失敗が続いたときの待ち時間。倍々に伸ばして上限で止める。"""
    return min(2.0**attempt, _BACKOFF_CAP)


def _count_motion(notifications: Any) -> int:
    """通知の束から「動いた」の数を数える。

    機種やファームで形が変わるので、読めないものは黙って飛ばす。ここで例外を出すと
    常駐タスクが死ぬ。
    """
    count = 0
    for note in notifications or []:
        try:
            topic = str(note.Topic._value_1)
            if _MOTION_TOPIC not in topic:
                continue
            for item in note.Message._value_1.Data.SimpleItem:
                if item.Name == "IsMotion" and str(item.Value).lower() == "true":
                    count += 1
        except Exception:  # noqa: BLE001, S112
            logger.debug("読めない通知を飛ばす")
            continue
    return count


class MotionEventWatcher:
    """動体イベントを購読し、動きがあるたびに `on_motion` を呼ぶ常駐タスク。"""

    def __init__(self, onvif_getter: Callable[[], Any],
                 on_motion: Callable[[], None]) -> None:
        self._onvif_getter = onvif_getter
        self._on_motion = on_motion
        self._task: asyncio.Task | None = None
        self._pull: Any = None

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="motion-events")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _wait_for(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    async def _subscribe(self) -> Any:
        """購読して PullPoint サービスを返す。"""
        onvif = self._onvif_getter()
        # ONVIF はまだ繋がっていないことがある（`_cam_onvif` は `_ensure_connected()` を
        # 呼ぶまで None）。繋いでから渡せるよう、待てる値も受け取る。
        if inspect.isawaitable(onvif):
            onvif = await onvif
        if onvif is None:
            return None
        events = await onvif.create_events_service()
        sub = await events.CreatePullPointSubscription({"InitialTerminationTime": _TERMINATION})
        ref = sub.SubscriptionReference.Address
        address = str(getattr(ref, "_value_1", None) or ref)
        # カメラが能力一覧に載せない宛先を手で入れる（実機の癖）。
        onvif.xaddrs[_PULLPOINT_NS] = address
        self._pull = await onvif.create_pullpoint_service()
        logger.info("動体イベントを購読した: %s", address)
        return self._pull

    async def _run(self) -> None:
        failures = 0
        while True:
            try:
                pull = await self._subscribe()
                if pull is None:
                    return                      # カメラが無い構成では黙って終わる
                failures = 0
                while True:
                    msg = await pull.PullMessages(
                        {"Timeout": _PULL_TIMEOUT, "MessageLimit": _PULL_LIMIT}
                    )
                    n = _count_motion(getattr(msg, "NotificationMessage", None))
                    if n:
                        logger.debug("動体イベント %d 件", n)
                        self._on_motion()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                wait = _backoff(failures)
                failures += 1
                # 切断は3回に2回起きる。異常ではないので debug に落とし、続くときだけ知らせる。
                logger.debug("動体イベントが切れた（%.0f秒後に取り直す）: %s", wait, e)
                if failures == 5:
                    logger.warning("動体イベントの購読が続けて失敗している: %s", e)
                await self._wait_for(wait)
