"""人ごとの認識埋め込みの、cosine 最大＋しきい値の純判定。

顔（ArcFace）と声（ECAPA-TDNN）で共用する。実モデルに依らない部分をここへ寄せ、テストしやすくする。
保存は PostgreSQL の `recognition_embeddings`（`store/recognition_embeddings.py`・知-ae）。以前の pickle は撤去した。
"""

from __future__ import annotations

import numpy as np


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
