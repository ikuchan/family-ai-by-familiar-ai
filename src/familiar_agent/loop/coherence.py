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


def facts_ctx(
    *, saw: bool, memories: "list", picture: bool = False, seen: "str | None" = None
) -> str:
    """この反復でループが知っていることを、そのまま並べる。

    `saw` は役割 `見た` が並びに載ったか（`see` の完了で `_note_record` が付ける）。
    `picture` は主LLM が**画像そのもの**を受け取ったか、`seen` は見た印の文（ラベル列）。
    軽量LLM は画像を持たないので、主LLM が画像を見て語ったことを「与えられていない」と
    誤って違反にしないよう、受け取った事実と写っていたものの名前を渡す（2026-09-12 実機で、
    人が2人見えると正しく語った返事を `no-fake-perception` で差し戻した）。
    `memories` は W に実際に載った記録で、落とされたものは含まない——載らなかった記憶は
    主LLM が見ていないので、それを材料と呼べない。
    """
    seen_line = (
        "はい（この求めで see を呼んだ）" if saw else "いいえ（この求めで see を呼んでいない）"
    )
    pic = "はい（主LLM は写真そのものを見ている）" if picture else "いいえ"
    if seen:
        pic += f"。印：{seen[:160]}"
    if not memories:
        mem = "0件（『昨日より』『前と違う』と言える材料は無い）"
    else:
        # 想起は口を通るので、受け取るのは `Recalled`（環-e-い）。日付は記録の時刻から作る。
        from ..store import clock

        dates = " / ".join(
            clock.ts_to_date(r.mi.timestamp) if r.mi.timestamp else "?" for r in memories
        )
        low = sum(1 for r in memories if r.confidence < CONF_UNCERTAIN)
        mem = f"{len(memories)}件（{dates}。うち conf<{CONF_UNCERTAIN} が{low}件）"
    return (
        f"[この反復で分かっていること]\n見たか：{seen_line}\n画像を受け取った：{pic}\n"
        f"作業状態に並んだ記憶：{mem}"
    )
