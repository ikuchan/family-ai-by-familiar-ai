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

from ..core.model_resource import ModelResource, load_local_first

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
    `None` を返す）。GPU に載るときは float16 で読む（環-ay・`根拠台帳` §72）。テストは並列にワーカーを起こし、**ワーカーごとに**モデルを載せるため、
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
        import torch

        # 手元の控えから読む（ネットがつながる前でも起きる・環-ax）。GPU では float16（環-ay）：GPU を約 1.07 GB
        # 空ける。本番の記録の本文 5,950 種で nan は 0 種、float32 との点の近さは最小 0.99972 だった（2026-11-21）。
        # 記録の点は float32 で作ってあるが、146 問の上位 7 件は平均 6.84 件が重なった。CPU は float32 のまま。
        # 載せる先の指定が無ければ None のまま渡す（ライブラリに任せる）。精度は GPU に載るかで決める。
        device = self.device
        on_gpu = device.startswith("cuda") if device else torch.cuda.is_available()
        kwargs = _gpu_model_kwargs("cuda" if on_gpu else "cpu", torch.float16)
        return load_local_first(
            self._model_name,
            lambda p: sentence_transformers.SentenceTransformer(
                p, device=device, model_kwargs=kwargs or None
            ),
        )

    def _encode(self, model: Any, texts: list[str]) -> "tuple[list[list[float]], list[bool]]":
        """点にして、float32 で長さ 1 にそろえる。数にならない点（float16 の溢れ）は 0 にして WARNING（環-ay）。

        返すのは点と、使える点かどうか。使えない点は控えに入れない（0 は、読めないときと同じ縮退の形）。
        """
        vecs = np.asarray(
            model.encode(texts, normalize_embeddings=False, show_progress_bar=False),
            dtype=np.float32,
        )
        norms = np.linalg.norm(vecs, axis=1)
        ok = np.isfinite(vecs).all(axis=1) & np.isfinite(norms) & (norms > 0)
        vecs[ok] /= norms[ok, None]
        vecs[~ok] = 0.0
        if not ok.all():
            logger.warning(
                "埋め込みが数にならなかった %d 件を 0 にした（float16 の溢れの疑い）",
                int((~ok).sum()),
            )
        return vecs.tolist(), ok.tolist()

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
                miss_idx.append(i)
                miss_texts.append(t)
                results.append(None)
        return miss_idx, miss_texts, results

    def _store(self, cache: OrderedDict, texts: list[str], vecs: list[list[float]]) -> None:
        for t, v in zip(texts, vecs):
            k = self._key(t)
            cache[k] = v
            cache.move_to_end(k)
        while len(cache) > self._CACHE_SIZE:
            cache.popitem(last=False)

    def encode_document(self, texts: list[str]) -> list[list[float]]:
        model = self.ensure()
        if model is None:
            return self._zeros(len(texts))
        miss_idx, miss_texts, results = self._lookup(self._d_cache, texts)
        if miss_texts:
            try:
                new, ok = self._encode(model, miss_texts)
            except Exception as e:
                logger.warning("encode_document failed: %s", e)
                return self._zeros(len(texts))
            good = [(t, v) for t, v, g in zip(miss_texts, new, ok) if g]
            self._store(self._d_cache, [t for t, _ in good], [v for _, v in good])
            for j, i in enumerate(miss_idx):
                results[i] = new[j]
        return results  # type: ignore

    def encode_query(self, texts: list[str]) -> list[list[float]]:
        model = self.ensure()
        if model is None:
            return self._zeros(len(texts))
        miss_idx, miss_texts, results = self._lookup(self._q_cache, texts)
        # 同じ文を文書として埋め込んだ控えがあれば使う（出-bz-い）。版を書いた直後の想起は、その版の文を手がかりに
        # するので、同じベクトルを 2 回計算していた（10/13 00:35・約 6,800 字）。2 つは同じ設定でモデルを呼ぶ。
        for i, t in zip(list(miss_idx), list(miss_texts)):
            vec = self._d_cache.get(self._key(t))
            if vec is not None:
                results[i] = vec
                miss_idx.remove(i)
                miss_texts.remove(t)
        if miss_texts:
            try:
                new, ok = self._encode(model, miss_texts)
            except Exception as e:
                logger.warning("encode_query failed: %s", e)
                return self._zeros(len(texts))
            good = [(t, v) for t, v, g in zip(miss_texts, new, ok) if g]
            self._store(self._q_cache, [t for t, _ in good], [v for _, v in good])
            for j, i in enumerate(miss_idx):
                results[i] = new[j]
        return results  # type: ignore


# ── 2 つめの頭 Ruri（記-y） ─────────────────────────────────────────────────

RURI_DIM = 768
_RURI_DOC_PREFIX = "検索文書: "
_RURI_QUERY_PREFIX = "検索クエリ: "


def _gpu_model_kwargs(device: str, dtype: Any) -> dict:
    """GPU に載せるときだけ、型を指定して直接読む（読んでから直すと、読み込みの一瞬だけ GPU を余計に使う）。CPU は fp32。"""
    return {"dtype": dtype} if device.startswith("cuda") else {}


def _ruri_model_kwargs(device: str) -> dict:
    """Ruri の読み込み方。GPU では **bf16**。fp16 では一部の記録が数にならない値（nan）になった（2026-11-03 実測）。"""
    import torch

    return _gpu_model_kwargs(device, torch.bfloat16)


class _RuriModel(ModelResource):
    """日本語の符号化器 Ruri（`cl-nagoya/ruri-v3-310m`・768 次元）。想起の 2 つめの頭（記-y）。

    **縮退の側**である。読み込めなくても止めず、Ruri は休んで bge-m3 だけで想起する。
    失敗したときは 0 ではなく **None** を返す。0 は「近さ 0」と区別できない値として表に残るため。

    数値にするのは**記録の本文**で、記録 1 件に 1 つ（面ごとに持たない）。平均は引かない
    （本人の決定・係数は平均を引かない近さで測った）。載せる先は bge-m3 と同じ `EMBEDDING_DEVICE`
    で選ぶ（2 つの頭を同じ所に置く）。
    """

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__(name="Ruri", device_env="EMBEDDING_DEVICE")
        if model_name is None:
            from ..config import MemoryConfig

            model_name = MemoryConfig().ruri_model
        self._model_name = model_name

    def _load(self) -> Any:
        for name in ("sentence_transformers", "huggingface_hub", "transformers"):
            logging.getLogger(name).setLevel(logging.ERROR)
        import sentence_transformers
        import torch

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        return load_local_first(
            self._model_name,
            lambda p: sentence_transformers.SentenceTransformer(
                p, device=device, model_kwargs=_ruri_model_kwargs(device)
            ),
        )

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]] | None:
        model = self.ensure()
        if model is None:
            return None
        try:
            vecs = model.encode(
                [prefix + t for t in texts], normalize_embeddings=True, show_progress_bar=False
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Ruri で数値にできなかった: %s", e, exc_info=True)
            return None
        arr = np.asarray(vecs, dtype=np.float32)
        if not np.isfinite(arr).all():
            logger.warning("Ruri の数値に数にならない値が混じったので使わない（%d 件）", len(texts))
            return None
        return arr.tolist()

    def encode_document(self, texts: list[str]) -> list[list[float]] | None:
        return self._encode(texts, _RURI_DOC_PREFIX)

    def encode_query(self, texts: list[str]) -> list[list[float]] | None:
        return self._encode(texts, _RURI_QUERY_PREFIX)


# ── MentalItem ──────────────────────────────────────────────────────────────
