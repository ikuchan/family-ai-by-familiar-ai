"""Face-based person recognition (InsightFace / ArcFace).

Register a person's face:
    from familiar_agent.recognition.face import register_face
    register_face(person_id, "/path/to/photo.jpg")

人ごとの ArcFace 埋め込みを PostgreSQL の `recognition_embeddings`（`kind='face'`・人の id キー）に持つ
（知-ae・2026-10-02。以前は pickle の人名キーだった）。照らすのは人ごとの重心。
実モデル（insightface + onnxruntime）は重いので遅延シングルトンで1回だけロードする。
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config import RecognitionConfig
from ..core.model_resource import ModelResource
from ..store.recognition_embeddings import RecognitionEmbeddingStore
from .embedding_store import best_match

if TYPE_CHECKING:
    from ..person_memory_manager import RecognitionHint

logger = logging.getLogger(__name__)

_FACE: "_FaceModel | None" = None  # プロセスで1つだけ持つ（読込が重い）


def _face_store() -> RecognitionEmbeddingStore:
    """共有接続の器（短く使う）。"""
    from ..db import get_db

    return RecognitionEmbeddingStore(get_db().conn())


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
    *,
    cfg: RecognitionConfig | None = None,
    store: RecognitionEmbeddingStore | None = None,
) -> "RecognitionHint | None":
    """画像から人を同定する。未登録・顔なし・モデル無しは None。

    特徴は人の id で持つので、人物表を引き直さない（以前は人名キーで `manager` から id を引いた・知-ae で改めた）。
    """
    cfg = cfg or RecognitionConfig()
    store = store or _face_store()
    return await asyncio.to_thread(_recognize_sync, image_path, cfg, store)


def _recognize_sync(
    image_path: str,
    cfg: RecognitionConfig,
    store: RecognitionEmbeddingStore,
) -> "RecognitionHint | None":
    from ..person_memory_manager import RecognitionHint

    emb = _extract_face_embedding(image_path, cfg)
    if emb is None:
        return None
    m = best_match(emb, store.centroids("face", "registered"), cfg.face_threshold)
    if m is None:
        return None
    person_id, score = m
    return RecognitionHint(
        person_id=person_id,
        confidence=max(0.0, min(1.0, score)),
        source="face",
        reason=f"arcface cos={score:.3f}",
    )


def register_face(
    person_id: str,
    image_path: str,
    *,
    cfg: RecognitionConfig | None = None,
    store: RecognitionEmbeddingStore | None = None,
) -> bool:
    """人 `person_id` の顔埋め込みを登録する（上限 `registered_max` を超えたら古いものから捨てる）。顔が取れなければ False。"""
    cfg = cfg or RecognitionConfig()
    store = store or _face_store()
    emb = _extract_face_embedding(image_path, cfg)
    if emb is None:
        logger.warning("顔が取れず登録できない: pid=%s image=%s", person_id, image_path)
        return False
    store.add(person_id, "face", "registered", emb, cap=cfg.registered_max)
    logger.info("顔を登録: %s", person_id)
    return True
