"""調べる前に、1 時間以内に答えた記録を記憶から探す（2026-11-23・本人の決定ア）。

11 月のログで、話しかけられてウェブ検索した 6 回のうち 2 回は、記憶に答えがあった（11/14 21:32 は 11 分前に同じ明日の天気を
調べて答えていた）。検索を投げる直前に、1 時間以内に閉じた最終版を Jev に見せ、どれがこの問いに答えているかを選ばせる。
選ばれたら検索せずにその記録で答える。範囲は天気も試合の予定も同じ 1 時間〔仮〕。
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.core import jev_judges
from familiar_agent.loop import finals

_FINAL = "〔最終版 2026-11-14 21:21:16〜21:21:53〕「明日の天気は」と聞かれ、明日は曇りのあと雨と調べて答えた"


def _client(pick, conf=0.9):
    c = MagicMock()
    c.available = True
    c.ask = AsyncMock(
        return_value=JevAnswer(ok=True, answers={"answers": {"choice": pick, "confidence": conf}})
    )
    return c


def _agent(rows, pick):
    a = MagicMock()
    a._oif.recent_finals = MagicMock(return_value=rows)
    a._jev = _client(pick)
    a.config.jev_confidence_min = 0.6
    a.config.jev_state_max_chars = 28_000
    a.config.memory.answer_reuse_window_sec = 3600
    return a


def test_a_recent_answer_is_reused():
    a = _agent([{"id": "f1", "content": _FINAL}], "f1")
    got = asyncio.run(finals.recent_answer(a, question="パジュー明日の天気は?", query="明日の天気"))
    assert got == _FINAL
    since, until = a._oif.recent_finals.call_args.args[:2]
    assert until - since == timedelta(seconds=3600)  # 1 時間より古い記録は候補にしない


def test_no_answer_among_them_means_search():
    """**反証の側**：どれも答えていなければ None（いまどおり調べる）。"""
    a = _agent([{"id": "f1", "content": _FINAL}], jev_judges.NOT_ANSWERED)
    assert (
        asyncio.run(finals.recent_answer(a, question="次の試合はいつ?", query="次の試合")) is None
    )


def test_no_recent_finals_means_no_question():
    a = _agent([], "f1")
    assert asyncio.run(finals.recent_answer(a, question="明日の天気は", query="明日の天気")) is None
    a._jev.ask.assert_not_awaited()


def test_low_confidence_means_search():
    a = _agent([{"id": "f1", "content": _FINAL}], "f1")
    a._jev = _client("f1", conf=0.3)
    assert asyncio.run(finals.recent_answer(a, question="明日の天気は", query="明日の天気")) is None


def _loop(monkeypatch, *, answer, kind="発話"):
    from familiar_agent.loop.event_loop import InformationProcessing
    from familiar_agent.loop.request import Request

    ip = InformationProcessing.__new__(InformationProcessing)
    ip._req = Request()
    ip._req.trigger_kind = kind
    ip._req.request_text = "パジュー明日の天気は?"
    ip._agent = MagicMock()
    ip._triggers = asyncio.Queue()
    ip._dif = MagicMock(camera=lambda: None, sensor=lambda: None)
    ip._dif.lookup = AsyncMock(return_value=("投げた", True))
    ip._confirm_blocks = lambda action: False
    asked = AsyncMock(return_value=answer)
    monkeypatch.setattr(finals, "recent_answer", asked)
    return ip, asked


def _run(ip):
    async def go():
        await ip._run_lookup_body(
            "search_deferred", {"query": "明日の天気"}, "明日の天気", "i-1", 1
        )
        return [ip._triggers.get_nowait() for _ in range(ip._triggers.qsize())]

    return asyncio.run(go())


def test_the_loop_answers_from_memory_without_searching(monkeypatch):
    ip, asked = _loop(monkeypatch, answer=_FINAL)
    got = _run(ip)
    ip._dif.lookup.assert_not_awaited()
    assert asked.await_args.kwargs["question"] == "パジュー明日の天気は?"
    assert got and got[0].kind == "完了" and _FINAL in got[0].result and not got[0].failed


def test_the_loop_searches_when_memory_has_no_answer(monkeypatch):
    """**反証の側**：記憶に答えが無ければ、いまどおり調べる。"""
    ip, _ = _loop(monkeypatch, answer=None)
    _run(ip)
    ip._dif.lookup.assert_awaited_once()


def test_only_spoken_requests_look_in_memory_first(monkeypatch):
    """情動が起点の調べもの（自分から調べに行く）は、いまどおり調べる（聞かれたことではない）。"""
    ip, asked = _loop(monkeypatch, answer=_FINAL, kind="情動")
    _run(ip)
    asked.assert_not_awaited()
    ip._dif.lookup.assert_awaited_once()


def test_an_answer_that_was_itself_from_memory_is_not_reused():
    """記憶で答えた回の最終版は候補にしない（9 時に調べた予報が、調べ直されないまま夕方まで渡り続けない）。"""
    reused = f"〔最終版〕「明日の天気は」と聞かれ、{finals.REUSED_MARK}60 分以内に調べて答えた記録：曇りのあと雨、と答えた"
    a = _agent([{"id": "f2", "content": reused}], "f2")
    assert asyncio.run(finals.recent_answer(a, question="明日の天気は", query="明日の天気")) is None
    a._jev.ask.assert_not_awaited()


def test_an_empty_question_does_not_look():
    a = _agent([{"id": "f1", "content": _FINAL}], "f1")
    assert asyncio.run(finals.recent_answer(a, question="", query="明日の天気")) is None
    a._oif.recent_finals.assert_not_called()


def test_an_answer_from_memory_is_not_counted_as_a_search(monkeypatch):
    """記憶で答えた調べものは「記憶を引いた」に付け替える（1 つの求めで 2 回までの検索に数えない）。"""
    from familiar_agent.loop.request import Lookup

    ip, _ = _loop(monkeypatch, answer=_FINAL)
    ip._agent.config.memory.answer_reuse_window_sec = 3600
    ip._req.lookups = [Lookup(index=1, action="search_deferred", query="明日の天気", generation=0)]
    got = _run(ip)
    assert ip._req.lookups[0].action == "recall"
    assert got[0].result.startswith(finals.REUSED_MARK) and "60 分以内" in got[0].result


def test_asking_to_look_again_is_listed_as_not_answered():
    """「調べ直して」「最新の」と頼まれたら、どれも答えていない側に入る（Jev への選択肢の説明に書いてある）。"""
    import inspect

    src = inspect.getsource(jev_judges.judge_answered)
    assert "調べ直して" in src and "最新の" in src
