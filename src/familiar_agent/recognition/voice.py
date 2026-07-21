"""Voice-based speaker identification (ECAPA-TDNN / speechbrain).

Register a person's voice:
    from familiar_agent.recognition.voice import VoiceIdentifier
    vi = VoiceIdentifier(manager)
    vi.register_voice(alice_id, "/path/to/sample.wav")

人ごとの ECAPA 埋め込みを ~/.familiar_ai/voice_embeddings.pkl に持つ（person_id キー）。
実モデル（speechbrain）は重いので遅延シングルトンで1回だけロードする。
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config import RecognitionConfig
from .embedding_store import EmbeddingStore, best_match

if TYPE_CHECKING:
    from ..person_memory_manager import PersonMemoryManager, RecognitionHint

logger = logging.getLogger(__name__)

VOICE_EMB_DB = Path.home() / ".familiar_ai" / "voice_embeddings.pkl"

_MODEL: Any = None          # speechbrain EncoderClassifier の遅延シングルトン


def _get_model() -> Any:
    """ECAPA-TDNN の話者埋め込みモデルを1回だけ構築する。失敗時は None。"""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        logger.debug("speechbrain 未インストール。話者同定を飛ばす")
        return None
    try:
        _MODEL = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        logger.info("ECAPA-TDNN ロード完了")
        return _MODEL
    except Exception as e:
        logger.warning("ECAPA-TDNN のロードに失敗（話者同定を無効化）: %s", e)
        return None


def _extract_voice_embedding(audio_path: str) -> np.ndarray | None:
    """音声から ECAPA 話者埋め込みを返す。失敗は None。"""
    model = _get_model()
    if model is None:
        return None
    try:
        import torchaudio

        signal, _sr = torchaudio.load(audio_path)
        emb = model.encode_batch(signal)
        return np.asarray(emb.squeeze().detach().cpu().numpy(), dtype=np.float32)
    except Exception as e:
        logger.warning("話者埋め込みの抽出に失敗: %s", e)
        return None


class VoiceIdentifier:
    def __init__(
        self,
        manager: "PersonMemoryManager",
        *,
        cfg: RecognitionConfig | None = None,
        store: EmbeddingStore | None = None,
    ) -> None:
        self._manager = manager
        self._cfg = cfg or RecognitionConfig()
        self._store = store or EmbeddingStore(VOICE_EMB_DB)

    def register_voice(self, person_id: str, audio_path: str) -> bool:
        """人 `person_id` の声埋め込みを登録する。取れなければ False。"""
        emb = _extract_voice_embedding(audio_path)
        if emb is None:
            logger.warning("声が取れず登録できない: pid=%s", person_id)
            return False
        self._store.save_embedding(person_id, emb)
        logger.info("声を登録: %s", person_id)
        return True

    async def identify_async(self, audio_path: str) -> "RecognitionHint | None":
        return await asyncio.to_thread(self._identify_sync, audio_path)

    def _identify_sync(self, audio_path: str) -> "RecognitionHint | None":
        from ..person_memory_manager import RecognitionHint

        emb = _extract_voice_embedding(audio_path)
        if emb is None:
            return None
        m = best_match(emb, self._store.get(), self._cfg.voice_threshold)
        if m is None:
            return None
        person_id, score = m
        return RecognitionHint(
            person_id=person_id,
            confidence=max(0.0, min(1.0, score)),
            source="voice",
            reason=f"ecapa cos={score:.3f}",
        )
