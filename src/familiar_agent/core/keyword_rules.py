"""語の軸と、軸ごとの順位による底上げ（記-k・2026-09-21）。純関数。

想起は埋め込みの近さと時刻の近さで候補を集めていた。**埋め込みは文全体の似かたを見る**ので、
問いの形（「覚えてる？」）に引かれ、中身の語（本・キャンプ）では引けない。実機 15:58：5 分前に
自分が話した「…キャンプの準備…本を更新…」は、時間の軸では 32 位で候補に入っていたのに、
採点では 11 位で 7 件に届かず、パジュは「思い出せない」と答えた。**取りこぼしは候補集めではなく
採点にあった。**

そこで**語の軸**を候補の軸として足し、軸ごとの順位で**底上げ**する。ある記録が軸で何位だったかを
`1/(k+順位)` に変えて足し、それを適合度に加えてから床と件数で絞る。いくつもの軸で上位に来た記録は
底上げがきつくなる。適合度の目盛りは残るので、5 軸（関連・時間・情動・根づき・在席）の効きは
変わらない（本人の決定：k=10）。実測では、探していた記録が 11 位 → 1 位になった。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: 使う名詞の細分。「副詞可能」（さっき・前・いま・明日）は**時を指すだけ**で中身を持たず、
#: 候補を膨らませるので使わない（時期は 5 軸の時間と `recall_when` が担う）。
KEEP_POS_GROUPS = frozenset({"一般", "固有名詞", "サ変接続"})
#: 順位を底上げに変える定数。小さいほど 1 位が強い（本人の決定・2026-09-21）。
BOOST_K = 10
#: これ以上の割合に当たる語は使わない（ありふれすぎて証拠にならない）。
DF_MAX_RATIO = 0.25


def pick_words(text: str) -> list[str]:
    """問いから中身の語（名詞）を取り出す。同じ語は 1 度だけ。読めなければ空。"""
    if not text:
        return []
    try:
        import pyopenjtalk
    except Exception:  # noqa: BLE001
        logger.debug("pyopenjtalk が無いので語の列は使わない")
        return []
    out: list[str] = []
    try:
        feats = pyopenjtalk.run_frontend(text)
    except Exception as e:  # noqa: BLE001
        logger.warning("語を切れなかったので語の列は使わない：%s", e)
        return []
    for f in feats:
        if f.get("pos") != "名詞" or f.get("pos_group1") not in KEEP_POS_GROUPS:
            continue
        word = str(f.get("string") or "").strip()
        if word and word not in out:
            out.append(word)
    return out


def drop_common(
    words: list[str], counts: dict, total: int, *, max_ratio: float = DF_MAX_RATIO
) -> list[str]:
    """ありふれすぎる語と、1 度も出てこない語を落とす。

    `counts` は語 → その語を含む記録の数。割合で見るのは、記録が増えても同じ線で効くため。
    """
    if not words or total <= 0:
        return []
    kept = []
    for w in words:
        n = int(counts.get(w, 0) or 0)
        if 0 < n and (n / total) < max_ratio:
            kept.append(w)
    return kept


def rank_boost(ranks, *, k: int = BOOST_K) -> float:
    """軸ごとの順位を底上げに変えて足す。`1/(k+順位)` の和。

    `ranks` はその記録が載った軸での順位（1 から数える）の並び。載っていない軸は含めない。
    どの軸にも載っていなければ 0——底上げが無いときは、いままでの採点そのままになる。
    """
    return sum(1.0 / (k + int(r)) for r in ranks)
