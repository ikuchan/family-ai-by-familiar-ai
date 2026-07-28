"""人が居るかの判定（YOLO）。在/不在の層で、誰かは問わない。

`知覚在席` §3-2 が定めるとおり、在/不在は **G（T 側・連続）が YOLO（person・GPU）で**担う。
全身ベースなので顔が見えない向きにも強く、**登録が要らない**。「誰か」は I 側の別の層が
必要時に InsightFace で解く（#17）。

判定はループの常駐部から繰り返し呼ばれるので、**例外を投げない**。読めない・推論できない
ときは「見えなかった」（0人）として返し、常駐タスクを殺さない。誤って「居る」と言うより
「見えなかった」と言うほうが安全である（滞留窓が誤検出を吸収する側に働く）。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# COCO の person。椅子や鞄まで数えると、部屋が常に「誰か居る」になる。
_PERSON_CLASS = 0
_DEFAULT_MODEL = "yolo11n.pt"


class PersonDetector:
    """フレーム1枚に人が何人写っているかを数える。

    モデルの読込は重いので初回の判定まで遅らせる（カメラが無い構成で起動を遅くしない）。
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: Any = None
        self._load_failed = False

    def _ensure_model(self) -> Any:
        if self._model is not None or self._load_failed:
            return self._model
        try:
            import ultralytics

            self._model = ultralytics.YOLO(self._model_name)
            logger.info("人検出のモデルを読み込んだ: %s", self._model_name)
        except Exception as e:  # noqa: BLE001
            # 重みが取れない環境でも、他の機能は動き続ける。二度目以降は試さない。
            logger.exception("人検出のモデルを読み込めない（在席は常に不在になる）: %s", e)
            self._load_failed = True
        return self._model

    def _count_sync(self, frame: Any) -> int:
        model = self._ensure_model()
        if model is None:
            return 0
        try:
            results = model.predict(frame, classes=[_PERSON_CLASS], verbose=False)
        except Exception as e:  # noqa: BLE001
            logger.exception("人検出に失敗したので見えなかったものとして扱う: %s", e)
            return 0
        return sum(len(r.boxes) for r in results)

    async def count(self, frame: Any) -> int:
        """フレーム（ファイルパスか配列）に写っている人の数。

        GPU の重い呼び出しなのでイベントループの外へ出す（`コード規約` の並行処理）。
        """
        return await asyncio.to_thread(self._count_sync, frame)
