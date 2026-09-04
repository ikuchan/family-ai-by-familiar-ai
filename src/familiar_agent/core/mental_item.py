"""MI（心的アイテム）の器と、観測行から器を組み立てる変換。

`PrimitiveMentalItem`（発火ペイロード PI の器＝emotion/drive）と、それを継承する
`MentalItem`（I 側の id/content/vector/supersedes/activation を加えた器）。純 dataclass で
副作用も loop 依存も持たない（境界R で `tools/memory.py` から切り出した・[D-MIモデル]）。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..mood_register import MoodPAD


@dataclass
class PrimitiveMentalItem:
    emotion: object | None = None   # PAD または未設定。A-1では未設定(None)
    drive: object | None = None     # 5欠乏 または未設定。A-1では未設定(None)


@dataclass
class MentalItem(PrimitiveMentalItem):
    id: str = ""
    content: str = ""
    vector: object | None = None
    supersedes: str | None = None
    activation: float | None = None
    #: 高ぶり。取り込みのときに I が内容の新規性から作る機械値で、T の「感じ＋欲」では
    #: ないので PI には置かない。感情が未測定でも残る（050）ため emotion とは別に持つ。
    arousal: float | None = None


def _row_to_mental_item(row) -> MentalItem:
    """観測行から MentalItem を組み立てる。

    行の PAD 列（emotion_p/pn/a/dom）が**4つそろって値を持つときだけ** `MoodPAD` を
    emotion に載せる。1つでも欠ければ未設定（`None`）である。`用語_略語一覧` の PI 項が
    「評価結果としての中立と、未評価の未設定とを区別するため」評価前は未設定で持つと
    定めており、欠けた軸を中立で埋めるとその区別が消える。

    以前は `row.get("emotion_p", 0.5)` で読んでいたが、`dict.get` は**キーがあって値が
    None なら None を返す**ので、既定は「列を SELECT していない」ときにしか効かない。
    050 が NULL を入れられるようにした時点から、「分からない」と「測っていない」を
    取り違えたまま `MoodPAD(p=None, ...)` を作っていた。

    A（高ぶり）は機械値で、感情が未測定でも残る（050）。捨てずに `arousal` へ載せる。
    drive・vector は後続で未設定。
    """
    axes = (
        row.get("emotion_p"), row.get("emotion_pn"),
        row.get("emotion_a"), row.get("emotion_dom"),
    )
    return MentalItem(
        id=row["id"],
        content=row["content"],
        supersedes=row["superseded_by"],
        # `importance` は P-1 で役目を失い 039 で落とした。値は 021 が
        # `groundedness_g0` へ移してある。(a0,n) からの導出は Phase 2。
        activation=row["groundedness_g0"],
        emotion=None if any(x is None for x in axes) else MoodPAD(*axes),
        arousal=row.get("emotion_a"),
        drive=None,
        vector=None,
    )
