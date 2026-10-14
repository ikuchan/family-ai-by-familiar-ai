"""最終版——求めを閉じるときに書く「聞かれて答えた」の塊（記-t 段 1・2026-10-14・本人の決定）。純関数。

それまでは「何を聞かれ、何を答えたか」が 1 件の記録になっていなかった。起点は人の言葉だけ、版は聞かれたことと調べた
結果で答えは入らず、答えの記録は言ったことだけだった。想起でどれか 1 件が上がっても、会話として意味をなさない。

最終版は版の鎖の最後に付く（親は起点・前の版は改訂で畳む）。文は機械が組む（判断は要らない）：

    〔最終版 2026-10-14 10:13:40〜10:14:05〕「明日の天気は？」と聞かれ、「明日は晴れだよ」と答えた。経緯：1番：…

**見分けは文の頭の印でする。** 向きは版と同じ `求め` で、`kind` は向きから決まるので使えない（記-u で別に考える）。
"""

from __future__ import annotations

from datetime import datetime, tzinfo

#: 最終版の印。文はこれで始まる（想起の頭「[そばに居た] 」などが前に付くことはある）。
MARK = "〔最終版 "


def head(start: datetime, end: datetime, tz: "tzinfo | None" = None) -> str:
    """時刻の頭。開始は起点の時刻、終了は閉じた時刻（生活時間＝ローカル）。"""
    s, e = start.astimezone(tz), end.astimezone(tz)
    return f"{MARK}{s:%Y-%m-%d %H:%M:%S}〜{e:%H:%M:%S}〕"


def _opening(trigger_kind: str, asked: str, added: "list[str]") -> str:
    """何が起きたか。人の言葉なら「聞かれ」、機器なら「起きて」、情動なら「湧いて」。"""
    if trigger_kind == "機器":
        return f"〔{asked}〕が起きて、"
    if trigger_kind == "情動":
        return f"〔{asked}〕が湧いて、"
    said = f"「{asked}」と聞かれ、"
    return said + "".join(f"「{a}」と言い足され、" for a in added if a)


def _answer(outcome: str, answer: str, block_reason: str) -> str:
    """どう閉じたか。声にならなかった言葉は、言ったことと取り違えないよう書き分ける。"""
    if outcome == "打ち切り":
        return "打ち切った"
    if outcome == "発話":
        return f"「{answer}」と答えた"
    if outcome == "独白":
        reason = f"（{block_reason}）" if block_reason else ""
        return f"「{answer}」と言おうとしたが、言えなかった{reason}"
    return f"声には出さず「{answer}」と思った" if answer else "何も言わなかった"


def body(
    *,
    trigger_kind: str,
    asked: str,
    added: "list[str]",
    outcome: str,
    answer: str,
    block_reason: str,
) -> str:
    """聞かれたことと答え（時刻の頭と経緯を除いた部分）。長すぎて切るときも、ここは必ず残す。"""
    return _opening(trigger_kind, asked, added) + _answer(outcome, answer, block_reason)


def course(steps: "list[str]") -> str:
    """経緯（調べた結果・見たもの）。無ければ空。"""
    return ("。経緯：" + "／".join(steps)) if steps else ""


def compose(
    *,
    trigger_kind: str,
    asked: str,
    added: "list[str]",
    outcome: str,
    answer: str,
    block_reason: str,
    steps: "list[str]",
    start: datetime,
    end: datetime,
    tz: "tzinfo | None" = None,
) -> str:
    """最終版の文。順は 時刻の頭 → 聞かれたこと → 答え → 経緯。"""
    return (
        head(start, end, tz)
        + body(
            trigger_kind=trigger_kind,
            asked=asked,
            added=added,
            outcome=outcome,
            answer=answer,
            block_reason=block_reason,
        )
        + course(steps)
    )


def cut(text: str, limit: int) -> str:
    """`limit` 字に収める。並びが 時刻の頭 → 聞かれたこと → 答え → 経緯 なので、切れるのは経緯の末尾。"""
    return text[:limit]


def is_final(text: str) -> bool:
    """最終版の文か（想起の頭「[そばに居た] 」などが前に付いていても見る）。"""
    return MARK in str(text or "")[:40]


def fit(current: str, candidates: "dict[str, str]", limit: int) -> "tuple[str, dict[str, str]]":
    """続き先の判定（記-t 段 2）へ送る文を、上限 `limit` 字に収める。

    収まっていればそのまま。超えれば、いまの最終版と候補の 1 件ずつを同じ字数まで切る（候補の数は減らさない・
    最終版は時刻の頭と聞かれたことが前にあるので、末尾を切っても何の塊かは残る）。50 字より短くはしない。
    """
    if len(current) + sum(len(v) for v in candidates.values()) <= limit:
        return current, dict(candidates)
    cap = max(50, limit // (len(candidates) + 1))
    return current[:cap], {k: v[:cap] for k, v in candidates.items()}
