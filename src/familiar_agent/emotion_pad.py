"""PAD↔感情ラベルの正本と PAD→ラベル逆引き（書き込み PAD 化 W2a）。

`LABEL_PAD` が12ラベル→4軸 PAD（快 P／不快 Pn／高ぶり A／支配 Dom・各 [0,1]）の
**生きた正本**。マイグレーション025 の `_LABEL_PAD` はこの値の**凍結写し**で、
移行を過去の一度きりの実行として再現する（両者の値一致はテストで固定）。

`label_from_pad` は PAD を最近傍（ユークリッド）で12ラベルへ量子化する。逆引きは
PAD を12点のどれかへ丸めるだけなので素朴なユークリッドで十分で、e 軸の logit 距離
(`_emotion_match`) は引き込まない（それは想起採点用で、モジュールも重い）。

この段（W2a）では `label_from_pad` は未接続で、実行時の呼び出しは W2b（評価器が
PAD を出し、消費者向けにラベルを派生させる段）で繋ぐ。
"""

from __future__ import annotations

from .mood_register import MoodPAD

# ラベル → (P, Pn, A, Dom)。emotion_pad が生きた正本、マイグレーション025 は凍結写し。
LABEL_PAD: dict[str, tuple[float, float, float, float]] = {
    "happy":     (0.80, 0.15, 0.55, 0.60),
    "excited":   (0.85, 0.15, 0.85, 0.65),
    "curious":   (0.60, 0.25, 0.65, 0.55),
    "moved":     (0.75, 0.50, 0.60, 0.45),
    "surprised": (0.50, 0.40, 0.85, 0.35),
    "nostalgic": (0.55, 0.55, 0.30, 0.45),
    "relieved":  (0.65, 0.20, 0.25, 0.60),
    "tender":    (0.70, 0.20, 0.35, 0.50),
    "playful":   (0.75, 0.15, 0.65, 0.65),
    "proud":     (0.75, 0.15, 0.55, 0.90),
    "sad":       (0.20, 0.75, 0.25, 0.30),
    "neutral":   (0.50, 0.50, 0.50, 0.50),
}


def label_from_pad(pad: MoodPAD) -> str:
    """PAD を最近傍（ユークリッド）の感情ラベルへ量子化する。

    `LABEL_PAD` の各点との二乗距離が最小のラベルを返す。同距離は辞書の並び順で
    決定的に決まる（先勝ち）。
    """
    point = (pad.p, pad.pn, pad.a, pad.dom)
    best_label = "neutral"
    best_d2 = float("inf")
    for label, ref in LABEL_PAD.items():
        d2 = sum((x - y) * (x - y) for x, y in zip(point, ref))
        if d2 < best_d2:
            best_d2 = d2
            best_label = label
    return best_label


# 感情軸の一次絞り用ベクトル（4次元）。
# e の距離 D²=Σ λ_i (logit(x_obs)-logit(x_mood))² は**ロジット空間の重み付きユークリッド
# 距離**なので、√λ を畳み込んだ点を持てば pgvector の L2 距離がそのまま D になる。
# λ を畳み込む以上、λ を変えたら格納値を一括で作り直す（案イ）。
# ε は活性導出・感情距離と共通の 0.001（ロジットの発散を避ける）。
_LOGIT_EPSILON = 0.001


def pad_to_search_vector(
    pad: tuple[float, float, float, float],
    *,
    lambdas: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    epsilon: float = _LOGIT_EPSILON,
) -> list[float]:
    """PAD を感情軸の検索空間（ロジット・√λ 畳み込み）へ写す。"""
    import math

    out: list[float] = []
    for x, lam in zip(pad, lambdas):
        x = min(max(float(x), epsilon), 1.0 - epsilon)
        out.append(math.sqrt(lam) * math.log(x / (1.0 - x)))
    return out
