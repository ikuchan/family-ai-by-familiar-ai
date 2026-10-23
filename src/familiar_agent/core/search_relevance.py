"""検索の返りが問いに合うかを照らす（知-an・2026-10-10・本人の決定イ）。

10/08 21:22「守谷市の明日の天気」で Tavily が Microsoft のページ 7 件だけを返し、成功としてそのまま主LLM へ渡った。
14 日分 156 件のうち 5 件（約 3%）が、問いと無関係な英語のページだけだった。結果 1 件ごとに「問いに答える材料か」を
Jev に聞き（1 回の呼び出しでまとめて）、「関係ない」を確信度 `min_conf` 以上で選んだ結果だけを捨てる。迷ったものは
残す——捨てすぎると、合っている答えまで失う。1 件も残らなければ None（呼び手がもう一方の検索で調べ直す）。
Jev が使えない・失敗した・段に分けられないときは、元の文をそのまま返す（照らさないのが倒し先）。

Brave の返り（題名・短い抜粋・リンク）は 3 択で照らす（知-bm・2026-10-24・本人の決定）：答えが書いてある／読めば
ありそう／関係ない。10/22 に、抜粋に答えが無くてもそのまま届いていたと分かった。「答えが書いてある」を確信度 `min_conf`
以上で選んだ結果が 1 件も無ければ、1 番目の選択が「読めばありそう」の結果から、その確率がいちばん高いページを読む先に
する（確率が無ければ確信度で代える）。迷ったものは答えと数えず、捨てもしない。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from ..backends.jev import choice
from .jev_judges import _ask, picked

logger = logging.getLogger(__name__)

RELEVANT = "材料になる"
UNRELATED = "関係ない"
_CRITERIA = {
    RELEVANT: "問いに答えるのに使える内容がある（同じ場所・同じ話題の情報）",
    UNRELATED: "問いと関係のない話題・場所・言語のページで、答えに使えない",
}
ANSWERED = "答えが書いてある"
MAYBE = "読めばありそう"
_CRITERIA_3 = {
    ANSWERED: "問いの答えそのものが、この抜粋に書いてある",
    MAYBE: "抜粋には答えが無いが、このページを読めば答えがありそう",
    UNRELATED: "問いと関係のない話題・場所・言語のページで、答えに使えない",
}
_EXCERPT = 300  # Jev に見せる本文の長さ（題名と URL のほかに）

_TITLE = re.compile(r"^Title: ", re.MULTILINE)
_URL = re.compile(r"^URL: (\S+)", re.MULTILINE)


@dataclass(frozen=True)
class Checked:
    """照らした後の文と、読みに行くページ（無ければ None）。"""

    text: str
    read: "str | None" = None


def split_results(text: str) -> "tuple[str, list[str]]":
    """`Title:` の行で、見出しと結果の段に分ける。段が無ければ（見出し＝全文, 空）。"""
    starts = [m.start() for m in _TITLE.finditer(text or "")]
    if not starts:
        return text, []
    head = text[: starts[0]]
    blocks = [text[a:b].rstrip("\n") for a, b in zip(starts, starts[1:] + [len(text)])]
    return head, blocks


def _excerpt(block: str) -> str:
    lines = block.splitlines()
    keep = [ln for ln in lines if ln.startswith(("Title: ", "URL: "))]
    body = " ".join(ln for ln in lines if not ln.startswith(("Title: ", "URL: ", "ID: ")))
    return "\n".join(keep + [body[:_EXCERPT]])


def _maybe_score(answer: Any, key: str) -> float:
    """「読めばありそう」の確率。1 番目の選択でなければ 0。確率が無ければ確信度で代える。"""
    got = (getattr(answer, "answers", None) or {}).get(key) or {}
    if got.get("choice") != MAYBE:
        return 0.0
    probs = dict(got.get("probabilities") or {})
    return float(probs.get(MAYBE, got.get("confidence", 0.0)) or 0.0)


async def keep_relevant(
    jev: Any, query: str, text: str, *, min_conf: float, three_way: bool = False
) -> "Checked | None":
    """問いに合わない結果を捨てた文。1 件も残らなければ None。照らせなければ元の文。

    `three_way`（Brave）なら、答えが 1 件も無いときに読みに行くページも選ぶ。
    """
    head, blocks = split_results(text)
    if not blocks or jev is None or not getattr(jev, "available", False):
        return Checked(text)
    criteria = _CRITERIA_3 if three_way else _CRITERIA
    state = f"[問い]\n{query}\n\n" + "\n\n".join(
        f"[結果 r{i}]\n{_excerpt(b)}" for i, b in enumerate(blocks)
    )
    questions = {
        f"r{i}": choice(f"検索結果 r{i} は、問い『{query}』に答える材料を含むか", criteria)
        for i in range(len(blocks))
    }
    answer = await _ask(jev, state, questions)
    if answer is None or not getattr(answer, "ok", False):
        return Checked(text)
    kept = [i for i in range(len(blocks)) if picked(answer, f"r{i}", min_conf) != UNRELATED]
    logger.info("検索の返りを照らした：%d 件中 %d 件が問いに合う", len(blocks), len(kept))
    if not kept:
        return None
    read = None
    if three_way and not any(picked(answer, f"r{i}", min_conf) == ANSWERED for i in kept):
        scored = [(_maybe_score(answer, f"r{i}"), i) for i in kept]
        best, at = max(scored, default=(0.0, -1))
        url = _URL.search(blocks[at]) if best > 0 else None
        read = url.group(1) if url else None
        logger.info("検索の抜粋に答えが無い → 読むページ：%s", read)
    return Checked(head + "\n\n".join(blocks[i] for i in kept), read)
