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


def _row_to_mental_item(row) -> MentalItem:
    """観測行から MentalItem を組み立てる。

    Y（W2a）：行の PAD 列（emotion_p/pn/a/dom）を `MoodPAD` として emotion に載せる。
    列を SELECT していない呼び出しでも `row.get` 既定0.5で中立になり安全。これで
    評価器の PAD・行の列・MI 器の emotion が同じ `MoodPAD` で一本化する（B-3 の
    tif.py が emotion に MoodPAD を使うのと型が揃う）。drive・vector は後続で未設定。
    """
    return MentalItem(
        id=row["id"],
        content=row["content"],
        supersedes=row["superseded_by"],
        # `importance` は P-1 で役目を失い 039 で落とした。値は 021 が
        # `groundedness_g0` へ移してある。(a0,n) からの導出は Phase 2。
        activation=row["groundedness_g0"],
        emotion=MoodPAD(
            p=row.get("emotion_p", 0.5),
            pn=row.get("emotion_pn", 0.5),
            a=row.get("emotion_a", 0.5),
            dom=row.get("emotion_dom", 0.5),
        ),
        drive=None,
        vector=None,
    )
