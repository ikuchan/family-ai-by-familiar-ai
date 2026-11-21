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

from ..core.model_resource import ModelResource, load_local_first

logger = logging.getLogger(__name__)

_MODEL_NAME = "facebook/dinov2-small"


class VisualEncoder(ModelResource):
    """画像1枚を 384 次元のベクトルにする。

    **縮退する側**である（`fatal=False`）——重みが取れなければ景色の驚きは出ないが、
    在席の判定（YOLO）は動き続ける。読み込みと失敗の扱いは モデル資源（MR）の型枠が持つ
    （出-c）。
    """

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        super().__init__(name="見えのエンコーダ", device_env="VISUAL_ENCODER_DEVICE")
        self._model_name = model_name
        self._processor: Any = None

    def _load(self) -> Any:
        import torch
        import transformers

        def _load_from(place: str) -> Any:
            processor = transformers.AutoImageProcessor.from_pretrained(place)
            # 型を指定して読む（出-dr）。指定しないと、読むときの torch の既定の型で作られる（ほかのモデルの読み込みが
            # 既定を bf16 にしていると、まるごと bf16 になる）。
            return processor, transformers.AutoModel.from_pretrained(place, dtype=torch.float32)

        # 手元の控えから読む（環-ax）。読むのは 1 つずつ（`load_local_first` の錠・出-dr）。
        self._processor, model = load_local_first(self._model_name, _load_from)
        return model.to(self._torch_device).eval()

    @property
    def _torch_device(self) -> str:
        """`transformers` は載せる先を明示しないと CPU のままなので、ここで決める。

        型枠の `device` は「指定が無ければ `None`＝ライブラリに任せる」だが、
        `model.to(None)` は使えない。**このモデルだけは自動判定まで自分で持つ。**
        """
        if self.device:
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"

    def _embed_sync(self, image_path: str) -> list[float] | None:
        import torch
        from PIL import Image

        with Image.open(image_path) as image:
            inputs = self._processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.ensure()(**inputs)
        return out.pooler_output[0].detach().cpu().tolist()

    def _dtype_note(self) -> str:
        """重みの型ごとの数と、いまの torch の既定の型（出-dr）。型が混ざって落ちたときに原因を確かめるため。"""
        try:
            import torch

            counts: dict[str, int] = {}
            model = self.ensure()
            for p in model.parameters() if model is not None else ():
                name = str(p.dtype).removeprefix("torch.")
                counts[name] = counts.get(name, 0) + 1
            weights = "・".join(f"{k} {v}" for k, v in sorted(counts.items())) or "なし"
            return (
                f"重み {weights}、既定の型 {str(torch.get_default_dtype()).removeprefix('torch.')}"
            )
        except Exception:  # noqa: BLE001
            return "型を読めない"

    def warm(self) -> None:
        """読み込むだけ（起動時の温め）。"""
        self.ensure()

    async def embed(self, image_path: str) -> list[float] | None:
        """画像を埋め込む。読めなければ `None`。

        GPU の重い呼び出しなのでイベントループの外へ出す（`コード規約` の並行処理）。
        """
        if self.ensure() is None:
            return None
        try:
            return await asyncio.to_thread(self._embed_sync, image_path)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception(
                "見えの埋め込みに失敗した（この回は見送る）: %s（%s）", e, self._dtype_note()
            )
            return None
