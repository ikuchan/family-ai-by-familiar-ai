"""声の特徴（ECAPA-TDNN / speechbrain）。

書き起こしの区切りの PCM から話者の特徴を取る（`embed_pcm`・`tools/local_stt` が呼ぶ）。照合と登録は
`loop/event_loop` が行い、人ごとの特徴は PostgreSQL の `recognition_embeddings`（`kind='voice'`）に持つ（知-ae）。
実モデル（speechbrain）は重いので遅延シングルトンで1回だけロードする。ファイルから照らす `VoiceIdentifier` は
どこからも呼ばれていなかったので撤去した（知-ae・2026-11-21）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)

_MODEL: Any = None
#: モデルの置き場（初めて使うときにダウンロードする。状態ではないので DB には置かない）。
_MODEL_DIR = (
    Path.home() / ".cache" / "familiar-ai" / "spkrec-ecapa-voxceleb"
)  # speechbrain EncoderClassifier の遅延シングルトン


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
        # CPU に置く（知-ae）：Whisper と GPU を取り合わない。ECAPA は小さく、1.5 秒の区切りなら CPU で足りる見込み。
        _MODEL = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(_MODEL_DIR),
            run_opts={"device": "cpu"},
        )
        logger.info("ECAPA-TDNN ロード完了")
        return _MODEL
    except Exception as e:
        logger.warning("ECAPA-TDNN のロードに失敗（話者同定を無効化）: %s", e)
        return None


def embed_pcm(pcm16le: bytes) -> np.ndarray | None:
    """16 kHz・16 bit の PCM から ECAPA の話者の特徴を返す（ファイルを介さない・知-ae）。取れなければ None。"""
    if not pcm16le:
        return None
    model = _get_model()
    if model is None:
        return None
    import torch

    samples = np.frombuffer(pcm16le, dtype="<i2").astype(np.float32) / 32768.0
    signal = torch.from_numpy(samples).unsqueeze(0)
    emb = model.encode_batch(signal)
    return np.asarray(emb.squeeze().detach().cpu().numpy(), dtype=np.float32)
