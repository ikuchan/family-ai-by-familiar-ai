"""発話前の検査へ渡す事実を組む（出-f）。

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

#: この求めで届いた結果（版・完了 O）を事実として渡すときの 1 件の上限（出-n）。O の上限は
#: 500 字だが完了 O は 8192 字まであるので、検索結果の数値が後ろにあっても入る幅を取る。
ARRIVED_CHARS = 2000


def used_lines(raw, w_id_map: "dict[str, str]", memories: "list") -> list[str]:
    """主LLM が `referred`／`important` と申告した記憶の中身（W の行の本文）を返す。

    照合は `apply_memory_verdicts` と同じ対応表（12 桁 → 完全な id）で行う。当たらない id は捨てる。
    """
    if not raw or not w_id_map:
        return []
    by_id = {r.mi.obs_id: r.mi.content for r in memories if getattr(r, "mi", None)}
    out: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        verdict = str(item.get("verdict", "")).strip().lower()
        if verdict not in ("referred", "important"):
            continue
        full = w_id_map.get(str(item.get("id", "")).replace("-", "")[:12])
        if full and full in by_id:
            out.append(by_id[full])
    return out


def facts_ctx(
    *,
    saw: bool,
    memories: "list",
    picture: bool = False,
    seen: "str | None" = None,
    used: "list[str] | None" = None,
    arrived: "list[str] | None" = None,
    confirming: str = "",
) -> str:
    """この反復でループが知っていることを、そのまま並べる。

    `saw` は役割 `見た` が並びに載ったか（`see` の完了で `_note_record` が付ける）。
    `picture` は主LLM が**画像そのもの**を受け取ったか、`seen` は見た印の文（ラベル列）。
    軽量LLM は画像を持たないので、主LLM が画像を見て語ったことを「与えられていない」と
    誤って違反にしないよう、受け取った事実と写っていたものの名前を渡す（2026-09-12 実機で、
    人が2人見えると正しく語った返事を `no-fake-perception` で差し戻した）。
    `memories` は W に実際に載った記録で、落とされたものは含まない——載らなかった記憶は
    主LLM が見ていないので、それを材料と呼べない。
    `used` は主LLM が `referred`／`important` と申告した記憶の**中身**（2026-09-13）。件数と
    日付だけでは、記憶にある天気で答えた返事が「検索も提示も無いのに事実を言った」
    （`no-invented-knowledge`）に見えた。根拠が作業状態にあることを、中身で示す。

    `arrived` はこの求めで**届いた結果**（open な版・完了 O の content・出-n）。主LLM が申告
    しなくても事実として渡す——届いたことはループが知っている。**切らない**（1 件
    `ARRIVED_CHARS` まで）。申告した記憶を 200 字で切っていたため、検索結果の「雨のち曇 ·
    最高 · 25 ℃」が見えず「事実に無い」と差し戻された（2026-09-13 15:55）。

    `confirming` はいまの確認待ちの枠（`agent.confirm_frame`・出-ag-ろ 穴 4）。規則
    `no-claim-while-confirming`（確認待ちのあいだは「掛けた」と言わない）を照らすのに要る。
    渡していなかったので、この規則だけはいつも判定できなかった。無いときも「なし」と書く。
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
    if used:
        leaned = "\n".join(f"  - {u[:ARRIVED_CHARS]}" for u in used)
        leaned_line = f"主LLM が使ったと申告した記憶（{len(used)}件・これが根拠）：\n{leaned}"
    else:
        leaned_line = "主LLM が使ったと申告した記憶：申告なし"
    if arrived:
        got = "\n".join(f"  - {a[:ARRIVED_CHARS]}" for a in arrived)
        arrived_line = f"\nこの求めで届いた結果（{len(arrived)}件・これも根拠）：\n{got}"
    else:
        arrived_line = ""
    confirm_line = (
        f"確認待ちの預かり：あり（まだ掛かっていない）——{confirming}"
        if confirming
        else "確認待ちの預かり：なし"
    )
    return (
        f"[この反復で分かっていること]\n見たか：{seen_line}\n画像を受け取った：{pic}\n"
        f"{confirm_line}\n"
        f"作業状態に並んだ記憶：{mem}\n{leaned_line}{arrived_line}"
    )
