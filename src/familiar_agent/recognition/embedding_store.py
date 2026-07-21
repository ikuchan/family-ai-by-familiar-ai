"""人ごとの認識埋め込みの保存と、cosine 最大＋しきい値の純判定。

顔（ArcFace）と声（ECAPA-TDNN）で共用する。実モデルに依らない部分をここへ寄せ、
テストしやすくする。保存は pickle（人ごとの正規化埋め込み）。
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def best_match(
    query: np.ndarray,
    enrolled: dict[str, np.ndarray],
    threshold: float,
) -> tuple[str, float] | None:
    """`enrolled` の中で query と cosine 最大の相手を返す（しきい値未満は None）。

    query か相手のノルムが 0、enrolled が空のときは None。返り値は (キー, cosine)。
    キーの意味（人名か person_id か）は呼び出し側が決める。
    """
    if not enrolled:
        return None
    q = np.asarray(query, dtype=np.float32).ravel()
    qn = float(np.linalg.norm(q))
    if qn == 0.0:
        return None
    best_key: str | None = None
    best_score = -1.0
    for key, ref in enrolled.items():
        r = np.asarray(ref, dtype=np.float32).ravel()
        rn = float(np.linalg.norm(r))
        if rn == 0.0:
            continue
        score = float(np.dot(q, r) / (qn * rn))
        if score > best_score:
            best_score, best_key = score, key
    if best_key is not None and best_score >= threshold:
        return best_key, best_score
    return None


class EmbeddingStore:
    """人ごとの埋め込みを pickle に持つ。顔・声で別ファイルにして使う。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._emb: dict[str, np.ndarray] = self._load()

    def _load(self) -> dict[str, np.ndarray]:
        if self._path.exists():
            try:
                with open(self._path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    return {k: np.asarray(v, dtype=np.float32) for k, v in data.items()}
            except Exception as e:
                logger.warning("埋め込みの読込に失敗（無視して空から始める）: %s", e)
        return {}

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            pickle.dump(self._emb, f)

    def save_embedding(self, key: str, vec: np.ndarray) -> None:
        self._emb[key] = np.asarray(vec, dtype=np.float32)
        self._persist()

    def get(self) -> dict[str, np.ndarray]:
        return self._emb
