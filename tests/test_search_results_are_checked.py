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
from familiar_agent.core.search_relevance import Checked, keep_relevant, split_results
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
    assert "Copilot" not in got.text  # 確かに関係ない
    assert "wikipedia" in got.text  # 迷ったものは残す
    assert "support.microsoft.com" in got.text
    assert got.read is None  # 2 択はページを読みに行かない


def test_nothing_left_is_none_and_no_jev_keeps_the_text():
    all_off = _jev(fill=("関係ない", 0.95))
    assert asyncio.run(keep_relevant(all_off, "守谷市の明日の天気", MS, min_conf=0.6)) is None
    off = _jev(available=False)
    assert asyncio.run(keep_relevant(off, "守谷市の明日の天気", MS, min_conf=0.6)) == Checked(MS)
    assert asyncio.run(keep_relevant(None, "守谷市の明日の天気", MS, min_conf=0.6)) == Checked(MS)


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
    async def judge(query, text, tool):
        return None if off_marker in text else Checked(text)

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


# ── Brave は 3 択で照らし、答えが無ければページを読む（知-bm・2026-10-24・本人の決定）─────────────────
#
# 10/22 に、説明文の「数字が要るか」では担い手を分けられないと分かった。Brave は題名・短い抜粋・リンクしか返さず、
# 抜粋に答えが無くてもそのまま届いていた。Brave の結果を「答えが書いてある／読めばありそう／関係ない」で照らし、
# 答えが 1 件も無ければ「読めばありそう」の確率がいちばん高いページを機械が読み、一覧と一緒に 1 回で届ける。

ANSWER, MAYBE, OFF = "答えが書いてある", "読めばありそう", "関係ない"
BRAVE = (
    "Title: 霜降とは\nDescription: 二十四節気の一つ\nURL: https://a.example/soukou\n\n"
    "Title: 霜降の意味と由来\nDescription: 霜が降りるころ\nURL: https://b.example/yurai\n\n"
    "Title: 守谷市の天気\nDescription: 晴れ\nURL: https://c.example/tenki"
)


def _jev3(picks: "dict[str, dict[str, float]]", *, with_probs: bool = True):
    """`picks` は問いの名前（r0, r1 …）→ 選択肢ごとの確率。1 番目の選択と確信度は確率から決める。"""
    jev = MagicMock()
    jev.available = True

    async def ask(state, questions):
        answers = {}
        for key in questions:
            probs = picks[key]
            top = max(probs, key=probs.get)
            got = {"choice": top, "confidence": probs[top]}
            if with_probs:
                got["probabilities"] = probs
            answers[key] = got
        return JevAnswer(ok=True, answers=answers)

    jev.ask = AsyncMock(side_effect=ask)
    return jev


def test_brave_is_asked_three_ways_and_the_likeliest_page_is_picked():
    jev = _jev3(
        {
            "r0": {ANSWER: 0.3, MAYBE: 0.5, OFF: 0.2},
            "r1": {ANSWER: 0.2, MAYBE: 0.7, OFF: 0.1},
            "r2": {ANSWER: 0.0, MAYBE: 0.05, OFF: 0.95},
        }
    )
    got = asyncio.run(keep_relevant(jev, "霜降の由来", BRAVE, min_conf=0.6, three_way=True))
    assert got is not None
    assert got.read == "https://b.example/yurai"  # 「読めばありそう」の確率が高いほう
    assert "c.example" not in got.text and "a.example" in got.text  # 関係ないものは除く
    criteria = jev.ask.await_args.args[1]["r0"]["criteria"]
    assert set(criteria) == {ANSWER, MAYBE, OFF}


def test_an_answer_in_the_list_means_no_page_is_read():
    """**反証の側**：答えが書いてある結果が 1 件でもあれば読みに行かない。迷った答えは答えと数えない。"""
    jev = _jev3(
        {
            "r0": {ANSWER: 0.8, MAYBE: 0.1, OFF: 0.1},
            "r1": {ANSWER: 0.1, MAYBE: 0.8, OFF: 0.1},
            "r2": {ANSWER: 0.0, MAYBE: 0.05, OFF: 0.95},
        }
    )
    got = asyncio.run(keep_relevant(jev, "霜降の由来", BRAVE, min_conf=0.6, three_way=True))
    assert got is not None and got.read is None
    doubtful = _jev3(
        {
            "r0": {ANSWER: 0.5, MAYBE: 0.4, OFF: 0.1},
            "r1": {ANSWER: 0.1, MAYBE: 0.8, OFF: 0.1},
            "r2": {ANSWER: 0.0, MAYBE: 0.05, OFF: 0.95},
        }
    )
    got = asyncio.run(keep_relevant(doubtful, "霜降の由来", BRAVE, min_conf=0.6, three_way=True))
    assert got is not None and got.read == "https://b.example/yurai"
    assert "a.example" in got.text  # 迷ったものは捨てもしない


def test_without_probabilities_the_confidence_ranks_the_pages():
    jev = _jev3(
        {
            "r0": {ANSWER: 0.2, MAYBE: 0.5, OFF: 0.3},
            "r1": {ANSWER: 0.1, MAYBE: 0.7, OFF: 0.2},
            "r2": {ANSWER: 0.0, MAYBE: 0.05, OFF: 0.95},
        },
        with_probs=False,
    )
    got = asyncio.run(keep_relevant(jev, "霜降の由来", BRAVE, min_conf=0.6, three_way=True))
    assert got is not None and got.read == "https://b.example/yurai"


def test_two_ways_never_reads_a_page():
    jev = _jev({"r0": ("関係ない", 0.9)})
    got = asyncio.run(keep_relevant(jev, "守谷市の明日の天気", MS, min_conf=0.6))
    assert got is not None and got.read is None
    assert set(jev.ask.await_args.args[1]["r1"]["criteria"]) == {"材料になる", "関係ない"}


def _reading_tool(results: "dict[str, CallResult]", checked: "Checked | None"):
    calls: list[tuple[str, dict]] = []
    tools_seen: list[str] = []

    async def search(tool, args):
        calls.append((tool, args))
        return results[tool]

    async def judge(query, text, tool):
        tools_seen.append(tool)
        return checked

    tool = DeferredSearchTool(search, judge=judge)
    delivered: list[tuple[str, bool]] = []
    tool.set_completion_sink(lambda q, r, *, failed=False: delivered.append((r, failed)))
    return tool, calls, tools_seen, delivered


PAGE = "霜降は、霜が降りはじめるころという意味。"


def test_the_picked_page_is_read_and_delivered_with_the_list():
    tool, calls, seen, delivered = _reading_tool(
        {"brave_web_search": CallResult(BRAVE, None, True), "fetch": CallResult(PAGE, None, True)},
        Checked(BRAVE, read="https://b.example/yurai"),
    )
    asyncio.run(tool._run("霜降の由来", "brave_web_search", "brave"))
    assert seen == ["brave_web_search"]
    assert calls[1] == ("fetch", {"url": "https://b.example/yurai"})  # 字数は fetch の既定
    assert len(delivered) == 1
    text, failed = delivered[0]
    assert failed is False
    assert text.startswith(BRAVE) and "[読んだページ] https://b.example/yurai" in text
    assert text.endswith(PAGE)


def test_no_page_to_read_means_no_fetch():
    """**反証の側**：読むページが無ければ fetch を呼ばない。"""
    tool, calls, _, delivered = _reading_tool(
        {"brave_web_search": CallResult(BRAVE, None, True)}, Checked(BRAVE)
    )
    asyncio.run(tool._run("霜降の由来", "brave_web_search", "brave"))
    assert [c[0] for c in calls] == ["brave_web_search"]
    assert delivered == [(BRAVE, False)]


def test_an_unreadable_page_still_delivers_the_list():
    for bad in (CallResult("timeout", None, False), RuntimeError("down")):
        results = {"brave_web_search": CallResult(BRAVE, None, True)}

        async def search(tool, args, bad=bad):
            if tool == "fetch":
                if isinstance(bad, Exception):
                    raise bad
                return bad
            return results[tool]

        async def judge(query, text, tool):
            return Checked(BRAVE, read="https://b.example/yurai")

        t = DeferredSearchTool(search, judge=judge)
        delivered: list = []
        t.set_completion_sink(lambda q, r, *, failed=False: delivered.append((r, failed)))
        asyncio.run(t._run("霜降の由来", "brave_web_search", "brave"))
        assert delivered == [(BRAVE, False)]  # 失敗としては届けない


def test_the_judge_is_told_the_tool_that_answered():
    """Brave が使えず Tavily に回ったら Tavily として照らす。調べ直しでは、調べ直した道具の名前を渡す。"""
    tool, _, seen, _ = _reading_tool(
        {
            "brave_web_search": CallResult("down", None, False),
            "tavily_search": CallResult(MS, None, True),
        },
        Checked(MS),
    )
    asyncio.run(tool._run("霜降の由来", "brave_web_search", "brave"))
    assert seen == ["tavily_search"]

    seen2: list[str] = []

    async def search(tool, args):
        return CallResult(MS if tool == "tavily_search" else WEATHER, None, True)

    async def judge(query, text, tool):
        seen2.append(tool)
        return None if "Microsoft" in text else Checked(text)

    t = DeferredSearchTool(search, judge=judge)
    t.set_completion_sink(lambda q, r, *, failed=False: None)
    asyncio.run(t._run("守谷市の明日の天気", "tavily_search", "tavily"))
    assert seen2 == ["tavily_search", "brave_web_search"]
