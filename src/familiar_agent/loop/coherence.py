"""整合チェックへ渡す事実を組む（出-f）。

**判定はここでしない。** 応答が規則に反するかは意味の判断で、軽量LLM の仕事である。
ここがするのは、その判断に要る**推測の要らない事実**を集めることだけである。

軽量LLM は応答と規則しか持たないと、「見ていないのに見たと言っている」かを判断できない。
見たかどうかを知らないからである。規則を渡していなかったときの捕捉が 3/18 で、渡したら
15〜18/18 になったのと同じ性質で（`根拠台帳` §25.3）、足りないのは判断力ではなく材料である。

**応答の文字列は受け取らない。** 語の表で文を落とせば、意味を読めない機械がパジュの
普通の発話を黙って消す。話者ゲートで「`description` でお願いするのではなく構造で落とす」
と決めたのは、機械にできることを機械にやらせるという意味であって、機械に意味を判断させる
という意味ではない。
"""

from __future__ import annotations

# 規則 `memory-evidence-confidence` が「仮説として扱う」と定める境目。
CONF_UNCERTAIN = 0.55


def facts_ctx(*, saw: bool, memories: list[dict]) -> str:
    """この反復でループが知っていることを、そのまま並べる。

    `saw` は役割 `見た` が並びに載ったか（`see` の完了で `_note_record` が付ける）。
    `memories` は W に実際に載った記録で、落とされたものは含まない——載らなかった記憶は
    主LLM が見ていないので、それを材料と呼べない。
    """
    seen = "はい（この反復で see を呼んだ）" if saw else "いいえ（この反復で see を呼んでいない）"
    if not memories:
        mem = "0件（『昨日より』『前と違う』と言える材料は無い）"
    else:
        dates = " / ".join(str(m.get("date", "?")) for m in memories)
        low = sum(1 for m in memories if float(m.get("confidence", 1.0)) < CONF_UNCERTAIN)
        mem = f"{len(memories)}件（{dates}。うち conf<{CONF_UNCERTAIN} が{low}件）"
    return f"[この反復で分かっていること]\n見たか：{seen}\n作業状態に並んだ記憶：{mem}"
