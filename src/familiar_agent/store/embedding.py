"""埋め込みモデルと、ベクトルの符号化・正規化・次元合わせ。

想起の採点でも書き込みでも使うが、SQL も想起ロジックも持たない。モデルの読み込みは
遅延（最初の呼び出しまで待つ）で、プロセス内でキャッシュする。

次元は `EMBEDDING_DIM` に固定する。モデルを変えたときに次元が変わると、既存の
ベクトルと比較できなくなるため、`_coerce_to_embedding_dim` で長さだけは必ず揃える。
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any

import numpy as np

from ..core.model_resource import ModelResource

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024


def _cosine_similarity(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    q = query / (np.linalg.norm(query) + 1e-10)
    c = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10)
    return c @ q


def _encode_vector(vec: list[float]) -> bytes:
    return np.array(vec, dtype=np.float32).tobytes()


def _decode_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def _coerce_to_embedding_dim(vec: np.ndarray) -> np.ndarray:
    """Ensure vec has exactly EMBEDDING_DIM dimensions (pad with zeros or truncate).

    In production embeddings are always EMBEDDING_DIM-dimensional, so this is a
    no-op. In tests that mock the encoder with a small vector the padding keeps
    situated-embedding storage and retrieval consistent.
    """
    if vec.shape[0] == EMBEDDING_DIM:
        return vec
    out = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    n = min(vec.shape[0], EMBEDDING_DIM)
    out[:n] = vec[:n]
    return out


# ── Lazy embedding model ───────────────────────────────────────────────────

class _EmbeddingModel(ModelResource):
    """言語の符号化器（bge-m3）。記憶に密着するので OIF の内側に置く。

    **致命の側**である（`fatal=False` のまま扱い、失敗は `failed` で外へ知らせる）。
    出-b が「埋め込みは致命」と決めているが、判定は呼び出し側（`memory.py`）が
    `failed` を見て行う形になっており、ここで例外を投げると起動そのものが壊れる。
    その形は変えない。

    載せる先は**未指定なら sentence-transformers の自動判定に任せる**（型枠の `device` が
    `None` を返す）。テストは並列にワーカーを起こし、**ワーカーごとに**モデルを載せるため、
    GPU では VRAM を使い切ってプロセスごと落ちる（実測で `CUDA out of memory`）。
    テスト側で `EMBEDDING_DEVICE=cpu` を選んで奪い合いを断つ。
    """

    _CACHE_SIZE = 128

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        super().__init__(name="埋め込み", device_env="EMBEDDING_DEVICE")
        self._model_name = model_name
        self._q_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._d_cache: OrderedDict[str, list[float]] = OrderedDict()

    def is_ready(self) -> bool:
        """読み込みが**済んだか**（成功・失敗を問わず、試し終えたか）。

        型枠の `ready` は「使える状態か」なので、失敗したときに偽になる。ここが表すのは
        「先読みが終わったか」で、意味が違うため別に持つ。
        """
        return self.ready or self.failed

    def _load(self) -> Any:
        for name in ("sentence_transformers", "huggingface_hub", "transformers"):
            logging.getLogger(name).setLevel(logging.ERROR)
        import sentence_transformers

        if self.device:
            return sentence_transformers.SentenceTransformer(
                self._model_name, device=self.device)
        return sentence_transformers.SentenceTransformer(self._model_name)

    def _zeros(self, n: int) -> list[list[float]]:
        return [[0.0] * EMBEDDING_DIM for _ in range(n)]

    def _key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _lookup(self, cache: OrderedDict, texts: list[str]):
        results, miss_idx, miss_texts = [], [], []
        for i, t in enumerate(texts):
            k = self._key(t)
            if k in cache:
                cache.move_to_end(k)
                results.append(cache[k])
            else:
                miss_idx.append(i); miss_texts.append(t); results.append(None)
        return miss_idx, miss_texts, results

    def _store(self, cache: OrderedDict, texts: list[str], vecs: list[list[float]]) -> None:
        for t, v in zip(texts, vecs):
            k = self._key(t); cache[k] = v; cache.move_to_end(k)
        while len(cache) > self._CACHE_SIZE:
            cache.popitem(last=False)

    def encode_document(self, texts: list[str]) -> list[list[float]]:
        model = self.ensure()
        if model is None:
            return self._zeros(len(texts))
        miss_idx, miss_texts, results = self._lookup(self._d_cache, texts)
        if miss_texts:
            try:
                new = model.encode(miss_texts, normalize_embeddings=True,
                                   show_progress_bar=False).tolist()
            except Exception as e:
                logger.warning("encode_document failed: %s", e); return self._zeros(len(texts))
            self._store(self._d_cache, miss_texts, new)
            for j, i in enumerate(miss_idx): results[i] = new[j]
        return results  # type: ignore

    def encode_query(self, texts: list[str]) -> list[list[float]]:
        model = self.ensure()
        if model is None:
            return self._zeros(len(texts))
        miss_idx, miss_texts, results = self._lookup(self._q_cache, texts)
        if miss_texts:
            try:
                new = model.encode(miss_texts, normalize_embeddings=True,
                                   show_progress_bar=False).tolist()
            except Exception as e:
                logger.warning("encode_query failed: %s", e); return self._zeros(len(texts))
            self._store(self._q_cache, miss_texts, new)
            for j, i in enumerate(miss_idx): results[i] = new[j]
        return results  # type: ignore


# ── MentalItem ──────────────────────────────────────────────────────────────
