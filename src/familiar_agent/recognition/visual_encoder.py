"""見えの埋め込み（DINOv2）。定点ごとの「普通」を作るための視覚エンコーダ。

`知覚在席` §3-2 が定める見えの層。テキストを介さず構造の変化に敏感なので、配置のずれや
明るさの違いのように**エンティティに現れない変化**を拾える（CLIP を採らない理由も同じ）。

`facebook/dinov2-small`（ViT-S/14・384次元・Apache-2.0）。`transformers` は既に入っており、
新しい依存は増えない。重みは初回に取得され、以後は取得先の cache に残る。

在席の確認と同じ常駐から呼ばれるので、**例外を投げない**。読めないときは `None` を返し、
その回の「普通」の更新を見送る。1回抜けても EMA は次で追いつく。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

_MODEL_NAME = "facebook/dinov2-small"


class VisualEncoder:
    """画像1枚を 384 次元のベクトルにする。"""

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: Any = None
        self._processor: Any = None
        self._load_failed = False

    def _ensure_model(self) -> bool:
        if self._model is not None or self._load_failed:
            return self._model is not None
        try:
            import torch
            import transformers

            self._processor = transformers.AutoImageProcessor.from_pretrained(self._model_name)
            model = transformers.AutoModel.from_pretrained(self._model_name)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = model.to(self._device).eval()
            logger.info("見えのエンコーダを読み込んだ: %s（%s）", self._model_name, self._device)
        except Exception as e:  # noqa: BLE001
            # 重みが取れない環境でも、在席の判定（YOLO）は動き続ける。二度目以降は試さない。
            logger.exception("見えのエンコーダを読み込めない（景色の驚きは出ない）: %s", e)
            self._load_failed = True
        return self._model is not None

    def _embed_sync(self, image_path: str) -> list[float] | None:
        import torch
        from PIL import Image

        with Image.open(image_path) as image:
            inputs = self._processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
        return out.pooler_output[0].detach().cpu().tolist()

    async def embed(self, image_path: str) -> list[float] | None:
        """画像を埋め込む。読めなければ `None`。

        GPU の重い呼び出しなのでイベントループの外へ出す（`コード規約` の並行処理）。
        """
        if not self._ensure_model():
            return None
        try:
            return await asyncio.to_thread(self._embed_sync, image_path)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("見えの埋め込みに失敗した（この回は見送る）: %s", e)
            return None
