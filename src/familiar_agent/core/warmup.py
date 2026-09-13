"""重いモデルを起動時に温める（2026-09-13）。

人検出（YOLO）と見えのエンコーダ（DINOv2）は最初の `see` まで読まれず、初回の see が
5.3 秒（読込 2.8 秒）かかって「5 秒超え」のつなぎまで出た。読込と最初の推論（YOLO は
融合処理を伴う）を起動時に背景で済ませておく。**落ちても起動は続ける**（温めは速さのため
のもので、機能ではない）。
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def warm_models(*, detector=None, encoder=None) -> None:
    """持っているモデルを順に温める（背景・GPU を触るのでスレッドへ逃がす）。"""
    for name, model in (("人検出", detector), ("見えのエンコーダ", encoder)):
        if model is None:
            continue
        try:
            await asyncio.to_thread(model.warm)
            logger.info("%sを温めた", name)
        except Exception as e:  # noqa: BLE001
            logger.warning("%sを温められなかった（初回だけ遅くなる）: %s", name, e)
