"""人検出モデルの推論は直列にする（2026-09-13 実機で露見）。

在席（`count`）と即席ラベル（`labels`）が同じ YOLO を別スレッドから同時に呼び、読込直後の
融合処理が競合して `'Conv' object has no attribute 'bn'` で落ちた。モデル資源の鍵は読込だけを
守っており、推論は守っていなかった。
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock

from familiar_agent.recognition.person_detector import PersonDetector


class _RaceyModel:
    """同時に 2 本入ったら壊れるモデルのふり。"""

    def __init__(self) -> None:
        self.inside = 0
        self.overlapped = False
        self.names = {0: "person"}
        self._lock = threading.Lock()

    def predict(self, frame, **kwargs):
        with self._lock:
            self.inside += 1
            if self.inside > 1:
                self.overlapped = True
        time.sleep(0.05)
        with self._lock:
            self.inside -= 1
        r = MagicMock()
        r.boxes = MagicMock()
        r.boxes.__len__ = lambda _self: 0
        r.boxes.cls.tolist = MagicMock(return_value=[])
        return [r]


def test_count_and_labels_never_run_at_the_same_time() -> None:
    m = _RaceyModel()
    d = PersonDetector()
    d._mr_model = m

    async def both():
        await asyncio.gather(d.count("/tmp/a.jpg"), d.labels("/tmp/a.jpg"), d.count("/tmp/a.jpg"))

    asyncio.run(both())
    assert not m.overlapped, "推論が同時に走った"
