"""声で話者を決める規則（知-ae 段 2・2026-10-02・`設計方針_声で話者を見分ける` v0.1）。純関数。

声が決めるのは**誰が話しているか**だけで、在席（居るか）はカメラが決める。呼び手は在席があるときだけここを使う。

- **続ける**：いまの話者に緩い閾値（`voice_threshold`・0.25）以上で似ていれば、その人のまま。声の揺れ
  （かすれ・離れて話した）で外れないため。
- **付け替える**：別の人に厳しい閾値（`voice_switch_threshold`・0.35）以上で、いまの話者より近ければ、その人へ。
  似た声（兄弟）への取り違えを防ぐため、付け替えだけ厳しくする（本人の提案・2026-10-02）。
- **分からない**：どちらでもなければ既定の人に戻す（本人の決定イ）。顔は使わない（本人の決定ア）。
- **いまの話者に声の基準が無い**ときは続ける——その人ではないとは言えない。`/speaker` で決めたばかりで名乗った
  ことのない人が、次の声ですぐ外れないため（改造方針で承認した解釈）。
- **誰の基準も無い**ときは照らせない（`skip`・何もしない）。

`sure` は厳しい閾値以上で当たった印で、呼び手はその発話の声を今日の声に足す（登録の声には足さない——推定が
日をまたいで残ると取り違えが積み重なる。登録を育てるのは名乗りだけ）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Verdict:
    action: str  # keep・switch・unknown・skip
    person_id: "str | None" = None
    score: float = 0.0
    sure: bool = False  # 厳しい閾値以上で当たった（今日の声に足す）


def _cos(a: np.ndarray, b: np.ndarray) -> "float | None":
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(a, b) / (na * nb))


def scores(
    voice: np.ndarray,
    registered: "dict[str, np.ndarray]",
    today: "dict[str, np.ndarray]",
) -> "dict[str, float]":
    """人ごとの似かた。登録の重心と今日の重心のうち、近いほう。"""
    v = np.asarray(voice, dtype=np.float32).ravel()
    out: dict[str, float] = {}
    for refs in (registered, today):
        for pid, ref in refs.items():
            c = _cos(v, np.asarray(ref, dtype=np.float32).ravel())
            if c is not None and c > out.get(pid, -2.0):
                out[pid] = c
    return out


def decide(
    current: "str | None", scores: "dict[str, float]", *, loose: float, strict: float
) -> Verdict:
    """いまの話者（既定の人なら None）と人ごとの似かたから、話者をどうするかを決める。"""
    if not scores:
        return Verdict("skip")
    best = max(scores, key=lambda pid: scores[pid])
    top = scores[best]
    if current is not None and current in scores:
        mine = scores[current]
        if best == current and mine >= loose:
            return Verdict("keep", current, mine, mine >= strict)
        if top >= strict:
            return Verdict("switch", best, top, True)
        if mine >= loose:
            return Verdict("keep", current, mine, False)
        return Verdict("unknown", None, top)
    if top >= strict:
        return Verdict("switch", best, top, True)
    if current is not None:
        return Verdict("keep", current, 0.0, False)  # いまの話者に基準が無い
    return Verdict("unknown", None, top)
