"""検索の返りが問いに合うかを照らす（知-an・2026-10-10・本人の決定イ）。

10/08 21:22「守谷市の明日の天気」で Tavily が Microsoft のページ 7 件だけを返し、成功としてそのまま主LLM へ渡った。
14 日分 156 件のうち 5 件（約 3%）が、問いと無関係な英語のページだけだった。結果 1 件ごとに「問いに答える材料か」を
Jev に聞き（1 回の呼び出しでまとめて）、「関係ない」を確信度 `min_conf` 以上で選んだ結果だけを捨てる。迷ったものは
残す——捨てすぎると、合っている答えまで失う。1 件も残らなければ None（呼び手がもう一方の検索で調べ直す）。
Jev が使えない・失敗した・段に分けられないときは、元の文をそのまま返す（照らさないのが倒し先）。
"""

from __future__ import annotations

import logging
import re
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
_EXCERPT = 300  # Jev に見せる本文の長さ（題名と URL のほかに）

_TITLE = re.compile(r"^Title: ", re.MULTILINE)


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


async def keep_relevant(jev: Any, query: str, text: str, *, min_conf: float) -> "str | None":
    """問いに合わない結果を捨てた文。1 件も残らなければ None。照らせなければ元の文。"""
    head, blocks = split_results(text)
    if not blocks or jev is None or not getattr(jev, "available", False):
        return text
    state = f"[問い]\n{query}\n\n" + "\n\n".join(
        f"[結果 r{i}]\n{_excerpt(b)}" for i, b in enumerate(blocks)
    )
    questions = {
        f"r{i}": choice(f"検索結果 r{i} は、問い『{query}』に答える材料を含むか", _CRITERIA)
        for i in range(len(blocks))
    }
    answer = await _ask(jev, state, questions)
    if answer is None or not getattr(answer, "ok", False):
        return text
    kept = [b for i, b in enumerate(blocks) if picked(answer, f"r{i}", min_conf) != UNRELATED]
    logger.info("検索の返りを照らした：%d 件中 %d 件が問いに合う", len(blocks), len(kept))
    if not kept:
        return None
    return head + "\n\n".join(kept)
