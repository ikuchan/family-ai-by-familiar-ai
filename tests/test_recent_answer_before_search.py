"""調べる前に、記憶の答えを探す（出-dv・2026-11-23・本人の決め方）。

11 月のログで、話しかけられてウェブ検索した 6 回のうち 2 回は、記憶に答えがあった（11/14 21:32 は 11 分前に同じ明日の天気を
調べて答えていた）。検索を投げる直前に、記憶の最終版（直近のものと、この求めの W に載ったもの）を Jev に見せ、どれがこの問いに
答えているかを選ばせる。選ばれた記録が、済んだ結果か、予報・予定で調べてからの経過がいまから出来事までより短ければ、検索せずに
その記録で答える（過去の結果はどれだけ古くても使い、未来のことは半分を過ぎたら調べ直す）。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.core import jev_judges
from familiar_agent.loop import finals

_FINAL = "〔最終版 2026-11-23 09:49:00〜09:50:00〕「明日の天気は」と聞かれ、調べて「11月24日（月）は晴れ」と答えた"
_NOW = datetime(2026, 11, 23, 1, 0, tzinfo=timezone.utc)  # 日本時間 10:00


def _answer(key, value, conf=0.9):
    return JevAnswer(ok=True, answers={key: {"choice": value, "confidence": conf}})


def _client(pick, kind="forecast", remain="lt1d", conf=0.9, remain_conf=None):
    """1 回目（どれが答えているか）と 2 回目（種類・残り）の返り。`remain_conf` は残りの確信だけを変える。"""
    c = MagicMock()
    c.available = True
    second = JevAnswer(
        ok=True,
        answers={
            "kind": {"choice": kind, "confidence": conf},
            "remain": {
                "choice": remain,
                "confidence": conf if remain_conf is None else remain_conf,
            },
        },
    )
    c.ask = AsyncMock(side_effect=[_answer("answers", pick, conf), second])
    return c


def _rec(obs_id, ago):
    return MagicMock(
        mi=MagicMock(id=obs_id, obs_id=obs_id, direction="最終版", timestamp=_NOW - ago)
    )


def _agent(bodies, pick, kind="forecast", remain="lt1d", recent=()):
    a = MagicMock()
    a._oif.recent_finals = MagicMock(return_value=list(recent))
    a._oif.bodies = MagicMock(side_effect=lambda ids: {i: bodies[i] for i in ids if i in bodies})
    a._jev = _client(pick, kind, remain)
    a.config.jev_confidence_min = 0.6
    a.config.jev_state_max_chars = 28_000
    a.config.memory.answer_recent_hours = 6.0
    a.config.memory.answer_candidates_max = 8
    a.config.memory.answer_remain_hours = {
        "passed": 0,
        "lt1h": 0,
        "lt1d": 1,
        "lt3d": 24,
        "lt1w": 72,
        "lt1m": 168,
        "later": 720,
    }
    return a


def _ask(a, recalled, question="明日の天気は?", query="明日の天気"):
    return asyncio.run(
        finals.recent_answer(a, question=question, query=query, recalled=recalled, now=_NOW)
    )


def test_the_same_question_eleven_minutes_later_is_answered_from_memory():
    """A（本物の Jev の試しで a・予報・1 日以内）：経過 11 分 < 残りの下限 1 時間なので記憶で答える。"""
    a = _agent({"f1": _FINAL}, "f1", "forecast", "lt1d")
    assert _ask(a, [_rec("f1", timedelta(minutes=11))]) == _FINAL


def test_a_schedule_checked_a_week_ago_is_looked_up_again():
    """D（予報・残り 1 週間以内）を 7 日前に調べた：経過 168 時間 ≥ 下限 72 時間なので調べ直す。"""
    a = _agent({"f1": _FINAL}, "f1", "forecast", "lt1w")
    assert _ask(a, [_rec("f1", timedelta(days=7))]) is None


def test_a_forecast_whose_event_has_passed_is_looked_up_again():
    a = _agent({"f1": _FINAL}, "f1", "forecast", "passed")
    assert _ask(a, [_rec("f1", timedelta(minutes=20))]) is None


def test_a_finished_result_is_reused_however_old():
    """C（結果・もう過ぎた）：終わった試合の結果は、何日前でも調べ直さない。"""
    a = _agent(
        {"f1": "〔最終版 2026-11-16〕試合は 2 対 1 で勝った（終わっている）"},
        "f1",
        "result",
        "passed",
    )
    assert _ask(a, [_rec("f1", timedelta(days=7))], question="11月16日の試合の結果は?") is not None


def test_something_still_in_progress_is_looked_up_again():
    """進行中（試合の途中経過）は調べ直す（仮）。"""
    a = _agent({"f1": _FINAL}, "f1", "ongoing", "lt1h")
    assert _ask(a, [_rec("f1", timedelta(minutes=5))]) is None


def test_not_answered_means_search():
    """**反証の側**：B・E・F・G（翌朝の別の日・途中だけ・調べ直して・関係ない）は「答えていない」で調べる。"""
    a = _agent({"f1": _FINAL}, jev_judges.NOT_ANSWERED)
    assert _ask(a, [_rec("f1", timedelta(minutes=5))]) is None
    assert a._jev.ask.await_count == 1  # 2 回目（種類・残り）は聞かない


def test_recent_finals_are_candidates_even_when_not_recalled():
    """続けて聞いたとき：W に載っていなくても、直近の最終版は候補になる。"""
    a = _agent(
        {"f9": _FINAL},
        "f9",
        "forecast",
        "lt1d",
        recent=[{"id": "f9", "timestamp": _NOW - timedelta(minutes=5)}],
    )
    assert _ask(a, []) == _FINAL


def test_no_candidates_means_no_question():
    a = _agent({}, "f1")
    assert _ask(a, []) is None
    a._jev.ask.assert_not_awaited()


def test_low_confidence_means_search():
    a = _agent({"f1": _FINAL}, "f1")
    a._jev = _client("f1", conf=0.3)
    assert _ask(a, [_rec("f1", timedelta(minutes=5))]) is None


def test_an_answer_that_was_itself_from_memory_is_not_reused():
    """記憶で答えた回の最終版は候補にしない。印は記録の本文で見る（面の文は書き直されていることがある）。"""
    reused = f"〔最終版〕「明日の天気は」と聞かれ、{finals.REUSED_MARK}前に調べて答えた記録：晴れ"
    a = _agent({"f2": reused}, "f2")
    assert _ask(a, [_rec("f2", timedelta(minutes=5))]) is None
    a._jev.ask.assert_not_awaited()


def test_a_folded_final_is_not_a_candidate():
    """改訂で畳まれた最終版は本文が引けない（`bodies` が `not_hidden` で落とす）ので候補にならない。"""
    a = _agent({}, "f1")
    assert _ask(a, [_rec("f1", timedelta(minutes=5))]) is None
    a._jev.ask.assert_not_awaited()


def test_an_empty_question_does_not_look():
    a = _agent({"f1": _FINAL}, "f1")
    assert _ask(a, [_rec("f1", timedelta(minutes=5))], question="") is None
    a._oif.recent_finals.assert_not_called()


def test_jev_is_told_the_local_time_now():
    """「明日」は記録の時刻といまで別の日を指しうるので、いまの時刻を最終版の頭と同じ地方時で渡す。"""
    a = _agent({"f1": _FINAL}, "f1")
    _ask(a, [_rec("f1", timedelta(minutes=5))])
    assert f"{_NOW.astimezone():%Y-%m-%d %H:%M}" in str(a._jev.ask.await_args_list[0])


def _candidates(a):
    """1 回目の問いで Jev に見せた記録の id（「どれも答えていない」を除く）。"""
    criteria = a._jev.ask.await_args_list[0].args[1]["answers"]["criteria"]
    return [k for k in criteria if k != jev_judges.NOT_ANSWERED]


def _sent_chars(call):
    """Jev へ送った字数（送る文＋問いの文＋選択肢のキーと説明）。"""
    state, questions = call.args
    return len(state) + sum(
        len(q["instructions"]) + sum(len(k) + len(v) for k, v in q["criteria"].items())
        for q in questions.values()
    )


def test_the_latest_final_is_a_candidate_even_when_w_is_full_of_finals():
    """続けて聞いたとき：W に最終版が上限の 8 件載っていても、直近の最終版（前の答え）は候補に入る。"""
    bodies = {f"w{i}": _FINAL for i in range(8)} | {"r0": _FINAL}
    a = _agent(bodies, "r0", recent=[{"id": "r0", "timestamp": _NOW - timedelta(minutes=5)}])
    assert _ask(a, [_rec(f"w{i}", timedelta(days=i + 1)) for i in range(8)]) == _FINAL
    assert "r0" in _candidates(a) and len(_candidates(a)) == 8


def test_an_old_final_on_w_is_a_candidate_even_when_many_finals_are_recent():
    """**反証の側**：直近の最終版が 20 件あっても、W に載った古い最終版（過去の結果）は候補に入る。

    本番の 14 日の最終版 279 件で、それぞれが閉じた時点から見た直前 6 時間の最終版は中央 8 件、8 件以上が 54%
    （2026-11-23 に数えた）。直近だけで上限を埋めると、そういう時点では、古い結果を運べる W の側が落ちる。
    """
    bodies = {f"r{i}": _FINAL for i in range(20)} | {"w0": _FINAL}
    recent = [{"id": f"r{i}", "timestamp": _NOW - timedelta(minutes=i + 1)} for i in range(20)]
    a = _agent(bodies, "w0", "result", "passed", recent=recent)
    assert _ask(a, [_rec("w0", timedelta(days=7))]) == _FINAL
    got = _candidates(a)
    assert "w0" in got and "r0" in got and len(got) == 8


def test_the_cap_counts_only_records_that_can_be_used():
    """上限は、記憶で答えた回と畳まれた記録を落としたあとに掛ける（先に切ると、使える記録が残らない）。"""
    reused = f"〔最終版〕「明日の天気は」と聞かれ、{finals.REUSED_MARK}前に調べて答えた記録：晴れ"
    bodies = {f"w{i}": reused for i in range(4)} | {"w8": _FINAL}  # w4〜w7 は畳まれていて本文が無い
    a = _agent(bodies, "w8")
    assert _ask(a, [_rec(f"w{i}", timedelta(minutes=i + 1)) for i in range(9)]) == _FINAL
    assert _candidates(a) == ["w8"]


def test_recent_finals_are_read_back_to_answer_recent_hours():
    """直近の最終版は、いまから `answer_recent_hours` 前までを読む。"""
    a = _agent({"f1": _FINAL}, "f1")
    a.config.memory.answer_recent_hours = 3.0
    _ask(a, [_rec("f1", timedelta(minutes=5))])
    since, until = a._oif.recent_finals.call_args.args
    assert since == _NOW - timedelta(hours=3) and until == _NOW


def test_a_finished_result_is_reused_even_when_the_remaining_time_is_unsure():
    """済んだ結果は残りを問わないので、種類の確信が高ければ、残りの確信が低くても記憶で答える。"""
    a = _agent({"f1": _FINAL}, "f1")
    a._jev = _client("f1", "result", "passed", remain_conf=0.3)
    assert _ask(a, [_rec("f1", timedelta(days=7))]) == _FINAL


def test_a_forecast_with_an_unsure_remaining_time_is_looked_up_again():
    """**反証の側**：予報・予定は残りで決めるので、残りの確信が低ければ調べ直す。"""
    a = _agent({"f1": _FINAL}, "f1")
    a._jev = _client("f1", "forecast", "lt1d", remain_conf=0.3)
    assert _ask(a, [_rec("f1", timedelta(minutes=11))]) is None


def test_what_is_sent_to_jev_fits_the_limit_with_the_headings_and_choices():
    """Jev へ送る文は、見出し・問いの文・選択肢の説明を足しても上限に収まる（候補の合計が上限ぎりぎりのとき）。"""
    a = _agent({"f1": "あ" * 13_900, "f2": "い" * 13_900}, "f1")
    assert _ask(a, [_rec("f1", timedelta(minutes=5)), _rec("f2", timedelta(minutes=6))])
    assert [_sent_chars(c) <= 28_000 for c in a._jev.ask.await_args_list] == [True, True]


def test_an_overlong_record_is_cut_for_jev_but_returned_whole():
    """種類と残りを聞くときも、記録を上限に合わせて切る。答えに使う本文は切らない。"""
    long = "あ" * 40_000
    a = _agent({"f1": long}, "f1")
    assert _ask(a, [_rec("f1", timedelta(minutes=5))]) == long
    assert [_sent_chars(c) <= 28_000 for c in a._jev.ask.await_args_list] == [True, True]


def test_recalling_leaves_what_is_on_w_with_the_request():
    """W に載った記録は、想起が求めに預ける（`Request.recalled`）。調べものは背景で走るので、そこから読む。"""
    from familiar_agent.loop import workspace
    from familiar_agent.loop.request import Request

    rec = _rec("f1", timedelta(minutes=5))
    rec.mi.content, rec.fit, rec.confidence = _FINAL, 0.5, 0.8
    oif = MagicMock(recall=AsyncMock(return_value=[rec]), actors=MagicMock(return_value={}))
    req = Request()
    asyncio.run(workspace.recall(oif, "明日の天気", weights=None, req=req))
    assert req.recalled == [rec]


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
    assert asked.await_args.kwargs["recalled"] is ip._req.recalled  # W に載った記録を渡す
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


def test_an_answer_from_memory_is_not_counted_as_a_search(monkeypatch):
    """記憶で答えた調べものは「記憶を引いた」に付け替える（1 つの求めで 2 回までの検索に数えない）。"""
    from familiar_agent.loop.request import Lookup

    ip, _ = _loop(monkeypatch, answer=_FINAL)
    ip._req.lookups = [Lookup(index=1, action="search_deferred", query="明日の天気", generation=0)]
    got = _run(ip)
    assert ip._req.lookups[0].action == "recall"
    assert got[0].result.startswith(finals.REUSED_MARK)


def test_asking_to_look_again_is_listed_as_not_answered():
    """「調べ直して」「最新の」と頼まれたら、どれも答えていない側に入る（Jev への選択肢の説明に書いてある）。"""
    import inspect

    src = inspect.getsource(jev_judges.judge_answered)
    assert "調べ直して" in src and "最新の" in src
