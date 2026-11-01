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

#: この求めで届いた結果（版・完了 O）を事実として渡すときの 1 件の上限（出-n）。版は調べもの 1 件ずつに分けて
#: 渡すので、1 件は検索の結果とページの本文（出-dg ろ・2026-11-01 本人の決定：10,000 字）。
ARRIVED_CHARS = 10000

#: 上限を超えて届いた結果を縮めるとき、これ以下の件は縮めず、長い件もこれより短くしない（出-dg ろ・2026-11-02 本人の決定）。
KEEP_CHARS = 400

_ARRIVED_HEAD = "この求めで届いた結果（{n}件・これも根拠）："


#: 検査と申告に渡す、W の過去の記憶 1 件の上限（記憶の申告の問いと同じ・記-r）。
PAST_CHARS = 300


def past_rows(
    memories: "list", w_id_map: "dict[str, str]", cap: int = PAST_CHARS
) -> "dict[str, str]":
    """W の過去の記憶の列（12 桁 → 中身・`PAST_CHARS` まで）。対応表（`verdict_map`）に無い記録は載せない。

    対応表は過去の列だけを持つので、直近のやりとりの枠に載った行はここに来ない（記-r）。発話前の検査には、
    これを事実として渡し、同じ呼び出しで記憶ごとの申告を聞く。
    """
    by_id = {r.mi.obs_id: r.mi.content for r in memories if getattr(r, "mi", None)}
    return {
        short: str(by_id[full])[:cap]
        for short, full in (w_id_map or {}).items()
        if full in by_id and by_id[full]
    }


def facts_ctx(
    *,
    saw: bool,
    memories: "list",
    picture: bool = False,
    seen: "str | None" = None,
    past: "list[str] | None" = None,
    arrived: "list[str] | None" = None,
) -> str:
    """この反復でループが知っていることを、そのまま並べる。

    `saw` は役割 `見た` が並びに載ったか（`see` の完了で `_note_record` が付ける）。
    `picture` は主LLM が**画像そのもの**を受け取ったか、`seen` は見た印の文（ラベル列）。
    軽量LLM は画像を持たないので、主LLM が画像を見て語ったことを「与えられていない」と
    誤って違反にしないよう、受け取った事実と写っていたものの名前を渡す（2026-09-12 実機で、
    人が2人見えると正しく語った返事を `no-fake-perception` で差し戻した）。
    `memories` は W に実際に載った記録で、落とされたものは含まない——載らなかった記憶は
    主LLM が見ていないので、それを材料と呼べない。
    `past` は作業状態の過去の記憶の**中身**（`past_rows` が 1 件 `PAST_CHARS` までに切ったもの・記-r）。件数と日付だけでは、記憶にある天気で
    答えた返事が「検索も提示も無いのに事実を言った」（`no-invented-knowledge`）に見えた（2026-09-13）。以前は主LLM が
    使ったと申告した記憶だけを渡していたが、申告は Jev が検査と同じ呼び出しで出すので、過去の列をまるごと渡す。

    `arrived` はこの求めで**届いた結果**（open な版・完了 O の content・出-n）。主LLM が申告
    しなくても事実として渡す——届いたことはループが知っている。**切らない**（1 件
    `ARRIVED_CHARS` まで）。申告した記憶を 200 字で切っていたため、検索結果の「雨のち曇 ·
    最高 · 25 ℃」が見えず「事実に無い」と差し戻された（2026-09-13 15:55）。
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
    if past:
        leaned = "\n".join(f"  - {u}" for u in past)
        leaned_line = (
            f"作業状態の過去の記憶の中身（{len(past)}件・返事の根拠になりうる）：\n{leaned}"
        )
    else:
        leaned_line = "作業状態の過去の記憶の中身：なし"
    if arrived:
        got = "\n".join(f"  - {a[:ARRIVED_CHARS]}" for a in arrived)
        arrived_line = f"\n{_ARRIVED_HEAD.format(n=len(arrived))}\n{got}"
    else:
        arrived_line = ""
    return (
        f"[この反復で分かっていること]\n見たか：{seen_line}\n画像を受け取った：{pic}\n"
        f"作業状態に並んだ記憶：{mem}\n{leaned_line}{arrived_line}"
    )


def fit_arrived(items: "list[str]", room: int) -> "list[str]":
    """届いた結果を、検査と申告の state の残り `room` 字（出-bw）に収める（出-dg ろ）。

    収まっていればそのまま。超えれば `KEEP_CHARS` 以下の件は残し、それより長い件だけを同じ割合で縮める
    （`KEEP_CHARS` より短くはしない。それでも超えれば Jev が断り、倒し先へ倒れる）。記憶の行（`PAST_CHARS`）は
    `KEEP_CHARS` より短いので、ここでは縮めない。
    """
    items = [a[:ARRIVED_CHARS] for a in items]
    # facts_ctx が足す見出しと行頭（前の改行・見出し・各件の「改行＋  - 」）。
    room -= 1 + len(_ARRIVED_HEAD.format(n=len(items))) + len("\n  - ") * len(items)
    total = sum(len(a) for a in items)
    short = sum(len(a) for a in items if len(a) <= KEEP_CHARS)
    if total <= room or total == short:
        return items
    ratio = max(0, room - short) / (total - short)
    return [a if len(a) <= KEEP_CHARS else a[: max(KEEP_CHARS, int(len(a) * ratio))] for a in items]
