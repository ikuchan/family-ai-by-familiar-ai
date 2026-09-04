"""旧目盛りで書かれた PAD を未測定へ戻す（案A）。

快と不快の目盛りを「0＝無い」に揃えた。`_EMOTION_PAD_PROMPT` が P と Pn について
「0 none」と「0.5 neutral」を同居させており、中立の出来事をどちらに置くかが決まって
いなかったためである（実測でモデル間が 0.37 開いた・根拠台帳 §25.4）。あわせて
`LABEL_PAD["neutral"]` と mood の戻り先を (0.10, 0.10, 0.50, 0.50) へ移した。

既存行の P/Pn/Dom は**機械的に変換できない**。旧値には二つの読みが混ざっており、その
混在こそが目盛りを直した理由だからである。変換規則を作れば、どちらの読みで書かれた値かを
当て推量することになる。

そこで 050 が定めた未測定の形へ戻す。`emotion_p`／`emotion_pn`／`emotion_dom` と
`emotion_vec` を NULL にし、ラベルは `neutral` にする。`by_emotion` は
`emotion_vec IS NOT NULL` で絞るので、未測定は感情軸から自然に外れる。

**A（高ぶり）は触らない。** 機械値で、内容の新規性から作る。評価器へ渡していないので
目盛りの変更と関係が無い（050 の「PAD が未測定でも A は入る」と同じ扱い）。

**測り直しはここでやらない。** マイグレーションは開発とテストのたびに走るので、軽量LLM を
呼ぶと API 鍵と課金と数分の待ちがテストに入り込む。測り直しは
`scripts/remeasure_emotion_pad.py` が本番に対して手で走らせる。走らせるまでのあいだ、
DB は「測っていない」で一貫している。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE observations SET "
            "  emotion_p = NULL, emotion_pn = NULL, emotion_dom = NULL, "
            "  emotion_vec = NULL, emotion = 'neutral' "
            "WHERE emotion_p IS NOT NULL "
            "   OR emotion_pn IS NOT NULL "
            "   OR emotion_dom IS NOT NULL "
            "   OR emotion_vec IS NOT NULL"
        )
