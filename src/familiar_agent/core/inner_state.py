"""内部状態（気分・欲求）を主LLM へ渡す言葉（情-f・2026-09-14）。

**気分は 4 軸 × 5 段。** 境目はその軸の実測分布の分位（p10／p30／p70／p90）で、等頻度なので
どの段も実際に出る。以前は 12 点の感情ラベル表（`emotion_pad.LABEL_PAD`）への最近傍 1 語で、
記-g で測り直した PAD の幅（P 0.10〜0.35・A 0.36〜0.50）は表の座標（P 0.85 など・記-g 前）と
別のスケールだったため、09-12〜14 の 15 回の更新が 15 回 `neutral` で情報量 0 だった。表と
`label_from_pad` は GUI と評価器のラベルに残す。プロンプトはここを使う。

**欲求は発火した軸を明示し、ほかはその軸の p70 を超えたものだけ弱く添える。** 固定の
0.5／0.75 では REST（人が居るあいだ放電されず 0.95 に張り付く）が常に「高」、BOND・ESTEEM
（0.00〜0.08）が常に「低」で、情報が無かった。軸ごとの分位なら、張り付いた軸は自分の p70 を
超えないので出ない。

境目（`MoodBands`・`DriveP70`）は層 3 の設定値で、REST 内省が等頻度を保つように合わせ直す
（`設計方針_REST内省_設定値を調整する`）。**生の数値はここから先へ出さない**（規則
`no-raw-internal-metrics`）。数値は計測ログにだけ書く。
"""

from __future__ import annotations

from dataclasses import dataclass

# 段の言葉。上から順に p90 以上／p70 以上／p30 以上／p10 以上／それ未満。
_LEVELS = ("とても高い", "高い", "ふつう", "低い", "とても低い")

#: 軸の日本語名（プロンプトに出る語）。
MOOD_AXES = (("p", "うれしさ"), ("pn", "つらさ"), ("a", "高ぶり"), ("dom", "余裕"))
DRIVE_AXES = (
    ("seeking", "SEEKING", "探索したい"),
    ("rest", "REST", "休みたい"),
    ("bond", "BOND", "つながりたい"),
    ("safety", "SAFETY", "安心したい"),
    ("esteem", "ESTEEM", "認められたい"),
)
_DRIVE_WEAK = {
    "seeking": "探索したさ",
    "rest": "休みたさ",
    "bond": "つながりたさ",
    "safety": "安心したさ",
    "esteem": "認められたさ",
}


@dataclass(frozen=True)
class MoodBands:
    """気分 4 軸の境目 (p10, p30, p70, p90)。初期値は 2026-09-14 の実測（記-g 後・30 日・1,646 件）。"""

    p: tuple[float, float, float, float] = (0.10, 0.10, 0.25, 0.35)
    pn: tuple[float, float, float, float] = (0.10, 0.10, 0.10, 0.15)
    a: tuple[float, float, float, float] = (0.36, 0.50, 0.50, 0.50)
    dom: tuple[float, float, float, float] = (0.50, 0.50, 0.55, 0.60)


@dataclass(frozen=True)
class DriveP70:
    """欲求 5 軸の p70。初期値は 2026-09-14 の実測（09-12〜14・2,093 回）。"""

    seeking: float = 0.66
    rest: float = 0.95
    bond: float = 0.03
    safety: float = 0.75
    esteem: float = 0.07


def level(value: float, bands: tuple[float, float, float, float]) -> str:
    """値を 5 段の言葉に。境目は上の段に入る（p90 以上＝とても高い …）。

    **分位が重なる軸は、その段を空にする。** 実測では A の p30・p70・p90 がすべて 0.50 で
    （値が 1 点に積み上がっている）、素朴に「p90 以上」とすると積み上がった値そのものが
    「とても高い」になる。積み上がった値は中央なので「ふつう」。上の段に入るのは、その境目が
    下の境目より真に大きいときだけ。
    """
    p10, p30, p70, p90 = bands
    if p90 > p70 and value >= p90:
        return _LEVELS[0]
    if p70 > p30 and value >= p70:
        return _LEVELS[1]
    if value < p10:
        return _LEVELS[4]
    if p30 > p10 and value < p30:
        return _LEVELS[3]
    return _LEVELS[2]


def mood_words(mood, bands: MoodBands) -> str:
    """気分を「うれしさ 高い・つらさ ふつう・…」に。未測定（None）ならそう言う。"""
    if mood is None:
        return "気分は測れていない"
    parts = [
        f"{name} {level(getattr(mood, axis), getattr(bands, axis))}" for axis, name in MOOD_AXES
    ]
    return "・".join(parts)


def drive_words(drives, fired: str, p70: DriveP70) -> str:
    """欲求を「発火：SEEKING（探索したい）（ほかに 安心したさ がやや）」に。"""
    head = "発火：なし"
    for axis, label, verb in DRIVE_AXES:
        if axis == (fired or "").lower():
            head = f"発火：{label}（{verb}）"
    weak = [
        _DRIVE_WEAK[axis]
        for axis, _label, _verb in DRIVE_AXES
        if axis != (fired or "").lower() and float(getattr(drives, axis, 0.0)) > getattr(p70, axis)
    ]
    if weak:
        return f"{head}（ほかに {'・'.join(weak)} がやや）"
    return head


def pi_line(mood, drives, *, fired: str, bands: MoodBands, p70: DriveP70) -> str:
    """主LLM へ渡す 1 行。生の数値を含まない。"""
    return f"[内部状態(PI)] 気分：{mood_words(mood, bands)} / {drive_words(drives, fired, p70)}"
