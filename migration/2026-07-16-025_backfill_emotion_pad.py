"""Backfill 既存行の感情 PAD を label→PAD 写像で埋める（書き込み PAD 化 W1b・移行専用）。

W1a（024）で追加した PAD 列は既存行が既定0.5のまま。この段（W1b）は、確定した
12ラベル→4軸 PAD の写像で既存行の PAD を更新する。**移行専用・一回限り**で、
実行時の機械写像 φ ではない。評価器が P/Pn/Dom を直接出力する W2 が入れば、
新規観測はこの表を通らない（W1b は W2 より前に一度だけ流れる前提）。

写像値は課題5 の PAD 定義（快 P／不快 Pn／喚起 A／支配 Dom・各 [0,1]・中立0.5）
に基づく確定値。両価（moved／nostalgic）は不快 Pn を上げ、proud は支配 Dom を
最大寄り、鎮静系（relieved／sad／nostalgic／tender）は喚起 A を低くしてある。
表に無いラベル（valid 外・過去の異常値）は既定0.5のまま（UPDATE 対象外）。
"""

# ラベル → (emotion_p, emotion_pn, emotion_a, emotion_dom)
_LABEL_PAD: dict[str, tuple[float, float, float, float]] = {
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


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        for label, (p, pn, a, dom) in _LABEL_PAD.items():
            cur.execute(
                "UPDATE observations "
                "SET emotion_p = %s, emotion_pn = %s, emotion_a = %s, emotion_dom = %s "
                "WHERE emotion = %s",
                (p, pn, a, dom, label),
            )
