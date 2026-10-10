"""検索の返りが問いに合うかを Jev が照らす（知-an・2026-10-10・本人の決定イ）。

10/08 21:22「守谷市の明日の天気」で Tavily が Microsoft のページ 7 件だけを返し、成功としてそのまま主LLM へ渡った。
14 日分 156 件のうち 5 件（約 3%）が、問いと無関係な英語のページだけだった（美容クリニック・処方薬・家庭向けでない
掲示板）。道具が失敗したときしか、もう一方へ切り替えていなかった。結果 1 件ごとに「問いに答える材料か」を Jev に
聞き、「関係ない」（確信度 0.6 以上）の結果は捨てる。全部外れたら、もう一方の検索で 1 回だけ調べ直す。届けるのは
調べ直した後の 1 回だけ（ループの調べものは届くまで待ちのままで、つなぎも止まらない）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.core.search_relevance import keep_relevant, split_results
from familiar_agent.mcp_client import CallResult
from familiar_agent.tools.deferred_search import DeferredSearchTool

MS = (
    "Detailed Results:\n\n"
    "Title: Microsoft 365 - Sign into Copilot\nURL: https://m365.cloud.microsoft\nContent: 45-1-44,P-R-1118122\n\n"
    "Title: Microsoft\nURL: https://en.wikipedia.org/wiki/Microsoft\nContent: Microsoft Corporation is\n\n"
    "Title: Sign in to Microsoft 365\nURL: https://support.microsoft.com/a\nContent: sign in"
)
WEATHER = (
    "Title: 守谷市の今日明日の天気 - tenki.jp\nURL: https://tenki.jp/x\nContent: 明日は晴れ 最高 26℃\n\n"
    "Title: 守谷市の天気 - Yahoo!天気\nURL: https://weather.yahoo.co.jp/x\nContent: 晴れ時々曇り"
)


def _jev(picks: "dict[str, tuple[str, float]] | None" = None, *, available=True, fill=None):
    """`picks` は問いの名前（r0, r1 …）→（選んだもの, 確信度）。`fill` は全部に同じ答え。"""
    jev = MagicMock()
    jev.available = available

    async def ask(state, questions):
        answers = {}
        for key in questions:
            choice, conf = (picks or {}).get(key, fill or ("材料になる", 0.9))
            answers[key] = {"choice": choice, "confidence": conf}
        return JevAnswer(ok=True, answers=answers)

    jev.ask = AsyncMock(side_effect=ask)
    return jev


def test_the_results_are_split_by_title():
    head, blocks = split_results(MS)
    assert head.strip() == "Detailed Results:"
    assert len(blocks) == 3 and blocks[0].startswith("Title: Microsoft 365")


def test_an_unrelated_result_is_dropped_but_a_doubtful_one_stays():
    jev = _jev({"r0": ("関係ない", 0.9), "r1": ("関係ない", 0.5)})
    got = asyncio.run(keep_relevant(jev, "守谷市の明日の天気", MS, min_conf=0.6))
    assert got is not None
    assert "Copilot" not in got  # 確かに関係ない
    assert "wikipedia" in got  # 迷ったものは残す
    assert "support.microsoft.com" in got


def test_nothing_left_is_none_and_no_jev_keeps_the_text():
    all_off = _jev(fill=("関係ない", 0.95))
    assert asyncio.run(keep_relevant(all_off, "守谷市の明日の天気", MS, min_conf=0.6)) is None
    off = _jev(available=False)
    assert asyncio.run(keep_relevant(off, "守谷市の明日の天気", MS, min_conf=0.6)) == MS
    assert asyncio.run(keep_relevant(None, "守谷市の明日の天気", MS, min_conf=0.6)) == MS


def _tool(results: "dict[str, str]", judge):
    calls: list[str] = []

    async def search(tool, args):
        calls.append(tool)
        return CallResult(results[tool], None, True)

    tool = DeferredSearchTool(search, judge=judge)
    delivered: list[tuple[str, bool]] = []
    tool.set_completion_sink(lambda q, r, *, failed=False: delivered.append((r, failed)))
    return tool, calls, delivered


def _judge_by_text(off_marker: str):
    async def judge(query, text):
        return None if off_marker in text else text

    return judge


def test_an_all_off_answer_is_searched_again_and_delivered_once():
    tool, calls, delivered = _tool(
        {"tavily_search": MS, "brave_web_search": WEATHER}, _judge_by_text("Microsoft")
    )
    asyncio.run(tool._run("守谷市の明日の天気", "tavily_search", "tavily"))
    assert calls == ["tavily_search", "brave_web_search"]
    assert delivered == [(WEATHER, False)]  # 外れた返りは届けない。届けるのは 1 回だけ


def test_both_off_is_a_failure():
    tool, calls, delivered = _tool(
        {"tavily_search": MS, "brave_web_search": MS}, _judge_by_text("Microsoft")
    )
    asyncio.run(tool._run("守谷市の明日の天気", "tavily_search", "tavily"))
    assert len(delivered) == 1
    text, failed = delivered[0]
    assert failed is True and "関係のある結果が見つからなかった" in text


def test_without_a_judge_nothing_changes():
    calls: list[str] = []

    async def search(tool, args):
        calls.append(tool)
        return CallResult(MS, None, True)

    tool = DeferredSearchTool(search)
    delivered: list = []
    tool.set_completion_sink(lambda q, r, *, failed=False: delivered.append((r, failed)))
    asyncio.run(tool._run("守谷市の明日の天気", "tavily_search", "tavily"))
    assert calls == ["tavily_search"] and delivered == [(MS, False)]
