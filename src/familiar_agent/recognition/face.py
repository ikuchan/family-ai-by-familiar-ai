"""Face-based person recognition (InsightFace / ArcFace).

Register a person's face:
    from familiar_agent.recognition.face import register_face
    register_face("alice", "/path/to/photo.jpg")

人ごとの ArcFace 埋め込みを ~/.familiar_ai/face_embeddings.pkl に持つ（人名キー）。
実モデル（insightface + onnxruntime）は重いので遅延シングルトンで1回だけロードする。
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config import RecognitionConfig
from ..core.model_resource import ModelResource
from .embedding_store import EmbeddingStore, best_match

if TYPE_CHECKING:
    from ..person_memory_manager import PersonMemoryManager, RecognitionHint

logger = logging.getLogger(__name__)

FACE_EMB_DB = Path.home() / ".familiar_ai" / "face_embeddings.pkl"

_STORE: EmbeddingStore | None = None
_FACE: "_FaceModel | None" = None    # プロセスで1つだけ持つ（読込が重い）


def _face_store() -> EmbeddingStore:
    global _STORE
    if _STORE is None:
        _STORE = EmbeddingStore(FACE_EMB_DB)
    return _STORE


class _FaceModel(ModelResource):
    """InsightFace の FaceAnalysis を持つ。

    **縮退する側**である（`fatal=False`）——顔が分からなくても、在/不在の判定（YOLO）と
    会話は動き続ける。**insightface が入っていない構成は永続的な失敗**として型枠が扱うので、
    呼ぶたびに読み直すことはない（出-c・以前は失敗を記憶していなかった）。
    """

    def __init__(self, cfg: RecognitionConfig) -> None:
        super().__init__(name="顔認識")
        self._cfg = cfg

    def _load(self) -> Any:
        import onnxruntime as ort

        from insightface.app import FaceAnalysis

        # nvidia pip ホイールの CUDA/cuDNN を先読みし、onnxruntime の CUDA プロバイダが
        # libcublasLt.so.12 等を見つけられるようにする。これが無いと provider の .so が
        # ロードできず、警告だけ出して黙って CPU に落ちる（torch は自前で preload する
        # ので影響を受けないが、onnxruntime は自動では load しない）。
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        app = FaceAnalysis(name=self._cfg.face_model, providers=self._cfg.provider_list())
        app.prepare(ctx_id=0)
        return app


def _get_model(cfg: RecognitionConfig) -> Any:
    """InsightFace の FaceAnalysis を1回だけ構築する。失敗時は None。"""
    global _FACE
    if _FACE is None:
        _FACE = _FaceModel(cfg)
    return _FACE.ensure()


def _extract_face_embedding(image_path: str, cfg: RecognitionConfig) -> np.ndarray | None:
    """画像から最大の顔の正規化 ArcFace 埋め込みを返す。顔が無い/失敗は None。"""
    model = _get_model(cfg)
    if model is None:
        return None
    try:
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            return None
        faces = model.get(img)
        if not faces:
            return None
        # 最大の顔を採る（bbox 面積）。
        face = max(
            faces,
            key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
        )
        return np.asarray(face.normed_embedding, dtype=np.float32)
    except Exception as e:
        logger.warning("顔埋め込みの抽出に失敗: %s", e)
        return None


async def recognize_face_async(
    image_path: str,
    manager: "PersonMemoryManager",
    *,
    cfg: RecognitionConfig | None = None,
    store: EmbeddingStore | None = None,
) -> "RecognitionHint | None":
    """画像から人を同定する。未登録・顔なし・モデル無しは None。"""
    cfg = cfg or RecognitionConfig()
    store = store or _face_store()
    return await asyncio.to_thread(_recognize_sync, image_path, manager, cfg, store)


def _recognize_sync(
    image_path: str,
    manager: "PersonMemoryManager",
    cfg: RecognitionConfig,
    store: EmbeddingStore,
) -> "RecognitionHint | None":
    from ..person_memory_manager import RecognitionHint

    emb = _extract_face_embedding(image_path, cfg)
    if emb is None:
        return None
    m = best_match(emb, store.get(), cfg.face_threshold)
    if m is None:
        return None
    name, score = m
    persons = {p["name"]: p for p in manager.list_persons()}
    if name not in persons:
        return None
    return RecognitionHint(
        person_id=persons[name]["id"],
        confidence=max(0.0, min(1.0, score)),
        source="face",
        reason=f"arcface cos={score:.3f}",
    )


def register_face(
    name: str,
    image_path: str,
    *,
    cfg: RecognitionConfig | None = None,
    store: EmbeddingStore | None = None,
) -> bool:
    """人 `name` の顔埋め込みを登録する。顔が取れなければ False。"""
    cfg = cfg or RecognitionConfig()
    store = store or _face_store()
    emb = _extract_face_embedding(image_path, cfg)
    if emb is None:
        logger.warning("顔が取れず登録できない: name=%s image=%s", name, image_path)
        return False
    store.save_embedding(name, emb)
    logger.info("顔を登録: %s", name)
    return True
