"""時間切れの答え（環-aw 第 1 段 項目 8・`設計方針_パジュの状態` v0.3 §3.4・2026-11-10 本人の決定）。

考えているの合計が 3 分に届いたとき、または主LLM の 1 回が 60 秒で時間切れになったとき、届いていた結果を下書きに
集めてから動いている仕事を止め、軽量LLM が「考えすぎてよく分からなくなった」を前置きに答えを書いて言い、求めを
閉じる。最終版には時間切れと書く。軽量LLM が失敗したら、クレジット切れと同じ形の録音を鳴らす。

主LLM の呼び出しが**例外**で終わったときは、いまのまま（空の返りで沈黙して閉じる）。電子音を鳴らせないときに送る
`limit_reached` では閉じない（耳を戻すだけ）。
"""

from __future__ import annotations

import asyncio
import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

from familiar_agent.backends.types import TurnResult
from familiar_agent.core import final_version as fv
from familiar_agent.loop import event_loop as el
from familiar_agent.loop.event_loop import Decision, Trigger
from familiar_agent.loop.request import Lookup
from tests._arbiter_fakes import writer_says

LOG = "familiar_agent.loop.event_loop"
ANSWER = "調べたら晴れでした"


def _ip(*, writer=None, wait=0.5):
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    a.config.timeout_answer_wait_sec = wait
    a._utility_backend = writer if writer is not None else writer_says(ANSWER)
    a._stance_context = lambda _stance: "立ち位置"
    ip = InformationProcessing(a)
    ip._req.request_id = "origin-1"
    ip._req.request_text = "明日の天気は？"
    ip._req.cue = "明日の天気は？"
    ip._req.trigger_kind = "発話"
    ip._dif.speak = AsyncMock()  # type: ignore[method-assign]
    ip._dif.say_recorded = AsyncMock(return_value=True)  # type: ignore[method-assign]
    ip._speech_check_violation = AsyncMock(return_value=(None, None))  # type: ignore[method-assign]
    return a, ip


def _finals(a):
    return [
        c.args[0]
        for c in a._memory.save_async_with_id.call_args_list
        if c.kwargs.get("direction") == fv.DIRECTION
    ]


def _answers(a):
    return [
        c.args[0]
        for c in a._memory.save_async_with_id.call_args_list
        if c.kwargs.get("direction") == "発話"
    ]


def _answer(ip, **kw):
    async def go():
        got = await ip._answer_timed_out(**kw)
        await ip.settle_finals()
        await ip.close()
        return got

    return asyncio.run(go())


def _prompt(a) -> str:
    return a._utility_backend.complete.await_args.args[0]


# ── 最終版の書き分け（純関数）────────────────────────────────────────────────────


def test_the_final_version_says_it_timed_out():
    body = fv.body(
        trigger_kind="発話",
        asked="明日の天気は？",
        added=[],
        outcome="発話",
        answer=ANSWER,
        block_reason="",
        timed_out=True,
    )
    assert body.endswith(f"時間切れになり、「{ANSWER}」と答えた")
    unsaid = fv.body(
        trigger_kind="発話",
        asked="明日の天気は？",
        added=[],
        outcome="沈黙",
        answer="",
        block_reason="",
        timed_out=True,
    )
    assert unsaid.endswith("時間切れになり、何も言わなかった")


def test_an_ordinary_close_does_not_say_it_timed_out():
    """**反証の側**：時間切れでなければ、いままでの書き方。"""
    body = fv.body(
        trigger_kind="発話",
        asked="明日の天気は？",
        added=[],
        outcome="発話",
        answer=ANSWER,
        block_reason="",
    )
    assert body.endswith(f"「{ANSWER}」と答えた") and "時間切れ" not in body


# ── 時間切れの答え ────────────────────────────────────────────────────────────


def test_it_answers_from_what_arrived_and_closes(caplog):
    a, ip = _ip()
    ip._req.lookups = [
        Lookup(index=1, action="search_deferred", query="天気", generation=0, result="晴れ 26℃")
    ]
    ip._req.draft = "明日は晴れかな"
    with caplog.at_level(logging.INFO, logger=LOG):
        spoken = _answer(ip, partial="ええと、晴れ")
    prompt = _prompt(a)
    assert "考えすぎてよく分からなくなった" in prompt  # 決定ム の前置き
    assert "明日の天気は？" in prompt and "晴れ 26℃" in prompt
    assert "明日は晴れかな" in prompt and "ええと、晴れ" in prompt
    assert spoken == ANSWER
    ip._dif.speak.assert_awaited_once()
    assert ip._dif.speak.await_args.args[0] == ANSWER
    assert _answers(a) == [f"自分が答えた：{ANSWER}"]  # 言った答えは答えとして残る
    assert f"時間切れになり、「{ANSWER}」と答えた。経緯：" in _finals(a)[0]  # 経緯は答えの後ろ
    assert ip._req.request_id is None  # 閉じた
    assert any("時間切れ=はい" in r.getMessage() for r in caplog.records)


def test_without_drafts_it_answers_from_the_words_alone():
    a, ip = _ip()
    _answer(ip)
    prompt = _prompt(a)
    assert "明日の天気は？" in prompt and "（なし）" in prompt


def test_the_work_in_flight_is_stopped_and_unread_results_do_not_leak():
    """届いていた結果は下書きに入れてから捨てる。動いている仕事は止める（決定 3）。"""
    a, ip = _ip()
    seen: dict = {}

    async def go():
        hang = asyncio.get_running_loop().create_task(asyncio.sleep(10))
        ip._background_tasks.add(hang)
        ip._req.lookups = [Lookup(index=1, action="search_deferred", query="天気", generation=0)]
        ip._drained_completions.append(Trigger(kind="完了", query="天気", result="くもり", index=1))
        await ip._answer_timed_out()
        await asyncio.sleep(0)
        seen["cancelled"] = hang.cancelled()
        await ip.settle_finals()
        await ip.close()

    asyncio.run(go())
    assert seen["cancelled"] is True
    assert "くもり" in _prompt(a)
    assert ip._drained_completions == [] and not ip._background_tasks


def test_a_violation_is_said_anyway_with_one_line(caplog):
    """発話前の検査が違反を見つけても言う（決定メ）。ログは決まった文と数だけ。"""
    a, ip = _ip()
    ip._speech_check_violation = AsyncMock(return_value=("捏造", None))  # type: ignore[method-assign]
    with caplog.at_level(logging.INFO, logger=LOG):
        _answer(ip)
    ip._speech_check_violation.assert_awaited_once()
    assert ip._dif.speak.await_args.args[0] == ANSWER
    lines = [r.getMessage() for r in caplog.records if "時間切れの答え" in r.getMessage()]
    assert any("違反" in m for m in lines)
    assert not any("捏造" in m for m in lines)  # 違反の中身（言葉を含みうる）は出さない


def _recorded(writer, **kw):
    a, ip = _ip(writer=writer, **kw)
    spoken = _answer(ip)
    return a, ip, spoken


def test_a_failing_writer_plays_the_recording():
    for writer in (
        writer_says(raises=RuntimeError("落ちた")),
        writer_says(""),
        writer_says(ANSWER, delay=0.5),
    ):
        a, ip, spoken = _recorded(writer, wait=0.05)
        ip._dif.say_recorded.assert_awaited_once()
        assert ip._dif.say_recorded.await_args.args[0] == el.TIMEOUT_RECORDING
        ip._dif.speak.assert_not_awaited()
        assert spoken == el.TIMEOUT_WORDS == "時間内に答えをまとめられませんでした。"
        assert _finals(a)[0].endswith(f"時間切れになり、「{el.TIMEOUT_WORDS}」と答えた")


def test_a_recording_that_cannot_play_closes_silently(caplog):
    """**反証の側**：録音を鳴らせなければ、答えたことにしない。"""
    a, ip = _ip(writer=writer_says(""))
    ip._dif.say_recorded = AsyncMock(return_value=False)  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING, logger=LOG):
        _answer(ip)
    assert _answers(a) == []
    # 決まった一言を、思ったことや言いたかったこととして残さない（黙って閉じる）
    assert _finals(a)[0].endswith("時間切れになり、何も言わなかった")
    assert any("録音" in r.getMessage() for r in caplog.records)


def test_the_writer_waits_up_to_the_configured_seconds():
    """**反証の側**：上限より早く返れば、録音ではなく書いた答えを言う。"""
    a, ip, spoken = _recorded(writer_says(ANSWER, delay=0.05), wait=0.5)
    assert spoken == ANSWER
    ip._dif.say_recorded.assert_not_awaited()


def test_a_closed_request_is_not_answered_again():
    a, ip = _ip()
    ip._req.request_id = None
    assert _answer(ip) == ""
    a._utility_backend.complete.assert_not_awaited()


# ── 3 分の入口 ─────────────────────────────────────────────────────────────────


def test_reaching_three_minutes_queues_a_time_out_for_the_request():
    a, ip = _ip()
    a.config.thinking_limit_sec = 0.1

    async def go():
        await ip._to_state("request_opened")
        await asyncio.sleep(0.3)
        got = []
        while not ip._triggers.empty():
            got.append(ip._triggers.get_nowait())
        ip._stop_beeps()
        await ip.close()
        return got

    got = asyncio.run(go())
    assert [(t.kind, t.query) for t in got] == [(el.TIMED_OUT_KIND, "origin-1")]


def test_a_beep_failure_does_not_queue_a_time_out():
    """**反証の側**：電子音を鳴らせないときの `limit_reached` は、耳を戻すだけ（決定 2）。"""
    a, ip = _ip()

    async def go():
        await ip._to_state("request_opened")
        await ip._to_state("limit_reached")
        await asyncio.sleep(0.05)
        empty = ip._triggers.empty()
        ip._stop_beeps()
        await ip.close()
        return empty

    assert asyncio.run(go()) is True


def test_the_driver_takes_the_time_out_before_completions():
    a, ip = _ip()
    t = Trigger(kind=el.TIMED_OUT_KIND, query="origin-1")
    done = Trigger(kind="完了", query="天気", result="晴れ", index=1)

    async def go():
        ip._triggers.put_nowait(done)
        ip._triggers.put_nowait(t)
        return await ip._take_trigger()

    assert asyncio.run(go()) is t
    assert ip._drained_completions == [done]  # 完了は完了箱に残り、下書きになる


def test_a_time_out_of_a_closed_request_is_dropped():
    a, ip = _ip()
    ip._answer_timed_out = AsyncMock(return_value="")  # type: ignore[method-assign]
    asyncio.run(ip._on_timed_out(Trigger(kind=el.TIMED_OUT_KIND, query="前の求め")))
    ip._answer_timed_out.assert_not_awaited()
    asyncio.run(ip._on_timed_out(Trigger(kind=el.TIMED_OUT_KIND, query="origin-1")))
    ip._answer_timed_out.assert_awaited_once()


# ── 主LLM の 60 秒 ────────────────────────────────────────────────────────────


def test_a_main_llm_time_out_is_marked_and_keeps_the_streamed_text():
    a, ip = _ip()
    a.config.main_llm_timeout_sec = 0.05

    async def hang(**kw):
        kw["on_text"]("晴れ")
        kw["on_text"]("かな")
        await asyncio.sleep(10)

    a.backend.stream_turn = hang
    result = asyncio.run(ip._main_llm_turn("sys", [], [], 100, None))
    assert result.stop_reason == el.MAIN_LLM_TIMED_OUT
    assert result.text == "晴れかな" and not result.tool_calls


def test_a_main_llm_failure_is_not_marked():
    """**反証の側**：例外はいまのまま（時間切れの印を付けない）。"""
    a, ip = _ip()

    async def boom(**_kw):
        raise RuntimeError("落ちた")

    a.backend.stream_turn = boom
    result = asyncio.run(ip._main_llm_turn("sys", [], [], 100, None))
    assert result.stop_reason != el.MAIN_LLM_TIMED_OUT


def _decision(result, *, retried=False, original_text=""):
    return Decision(
        result=result,
        memories=[],
        w_id_map={},
        verdict_view="__self__",
        recent_frame="",
        system=("安定", "可変"),
        effort="high",
        capped=False,
        retried=retried,
        original_text=original_text,
    )


def test_a_timed_out_decision_goes_to_the_time_out_answer():
    a, ip = _ip()
    ip._answer_timed_out = AsyncMock(return_value="答え")  # type: ignore[method-assign]
    d = _decision(TurnResult(stop_reason=el.MAIN_LLM_TIMED_OUT, text="途中"))
    assert asyncio.run(ip._act_on_decision(d, utterance="明日の天気は？", gen=0)) == "答え"
    assert ip._answer_timed_out.await_args.kwargs == {"partial": "途中", "decision": d}


def test_an_empty_decision_still_closes_silently():
    """**反証の側**：例外の空の返りは、時間切れの答えへ行かない。"""
    a, ip = _ip()
    ip._answer_timed_out = AsyncMock(return_value="答え")  # type: ignore[method-assign]
    ip._finish = AsyncMock()  # type: ignore[method-assign]
    d = _decision(TurnResult(stop_reason="end_turn", text=""))
    asyncio.run(ip._act_on_decision(d, utterance="明日の天気は？", gen=0))
    ip._answer_timed_out.assert_not_awaited()


def test_a_timed_out_retry_says_the_first_answer():
    """言い直しが時間切れなら、元の答えを言う（いままでの「言い直しが say を返さなかった」と同じ）。"""
    a, ip = _ip()
    ip._answer_timed_out = AsyncMock(return_value="答え")  # type: ignore[method-assign]
    ip._finish = AsyncMock()  # type: ignore[method-assign]
    d = _decision(
        TurnResult(stop_reason=el.MAIN_LLM_TIMED_OUT, text=""),
        retried=True,
        original_text="元の答え",
    )
    asyncio.run(ip._act_on_decision(d, utterance="明日の天気は？", gen=0))
    ip._answer_timed_out.assert_not_awaited()
    assert ip._dif.speak.await_args.args[0] == "元の答え"


# ── 録音の口 ───────────────────────────────────────────────────────────────────


def test_the_recording_goes_through_the_same_mouth():
    """録音も `_say_aloud` を通る（言い始めと言い終わりを状態図へ送る）。"""
    a, ip = _ip()

    async def go():
        await ip._to_state("request_opened")
        heads = []

        async def play(name, *, gain):
            heads.append(ip._paju().head_state())
            return True

        ip._dif.say_recorded = play  # type: ignore[method-assign]
        played = await ip._say_aloud(el.TIMEOUT_WORDS, recording=el.TIMEOUT_RECORDING)
        heads.append(ip._paju().head_state())
        ip._stop_beeps()
        await ip.close()
        return played, heads

    played, heads = asyncio.run(go())
    assert played is True
    assert heads == ["answering", "thinking"]  # 言っている → 書き留めている
    ip._dif.speak.assert_not_awaited()


def test_the_dif_plays_a_recording_and_reports_a_missing_one(tmp_path, caplog):
    from familiar_agent.io import dif as dif_mod

    d = dif_mod.DIF.__new__(dif_mod.DIF)
    d._cue_task = None
    d._music_ducker = None
    (tmp_path / "timeout_answer.wav").write_bytes(b"x")
    with (
        patch.object(dif_mod, "_SOUNDS", tmp_path),
        patch.object(dif_mod, "_play_wav", AsyncMock(return_value=True)) as play,
        patch("familiar_agent.core.reaction_cue.voiced", lambda p: p),
    ):
        assert asyncio.run(d.say_recorded("timeout_answer", gain=0.5)) is True
        play.assert_awaited_once_with(tmp_path / "timeout_answer.wav", 0.5)
        with caplog.at_level(logging.WARNING):
            assert asyncio.run(d.say_recorded("無い", gain=0.5)) is False
    assert any("無い.wav" in r.getMessage() for r in caplog.records)


def test_the_recording_ships():
    """録音は元の声と、Gemini の声ごとのフォルダにある（クレジット切れと同じ形）。"""
    from familiar_agent.io.dif import _SOUNDS

    assert (_SOUNDS / f"{el.TIMEOUT_RECORDING}.wav").exists()
    for voice in ("Laomedeia", "Puck", "ja-jp-commercial-3", "ja-jp-csagent-8"):
        assert (_SOUNDS / "voices" / voice / f"{el.TIMEOUT_RECORDING}.wav").exists(), voice


def test_the_wait_comes_from_config():
    from familiar_agent.config import AgentConfig

    with patch.dict(os.environ, {}, clear=True):
        wait = AgentConfig().timeout_answer_wait_sec
    assert wait == 5.0  # 本人の決定（内省の調整に入れない）


# ── 見直し 1 回目で足した（2026-11-10） ─────────────────────────────────


def test_a_late_result_of_another_request_is_not_a_draft():
    """**反証の側**：この求めの意図に当たらない完了（打ち切った前の求めの遅れた結果）は、下書きに混ぜない。"""
    a, ip = _ip()
    ip._drained_completions.append(Trigger(kind="完了", query="前の求めの語", result="別の話"))
    _answer(ip)
    assert "別の話" not in _prompt(a)


def test_the_closest_drafts_come_first():
    """長い結果で切れても、言おうとしていたことと途中の文は残る（前に並べる）。"""
    a, ip = _ip()
    a.config.completion_content_max = 200
    ip._req.lookups = [
        Lookup(index=1, action="search_deferred", query="天気", generation=0, result="長い" * 500)
    ]
    ip._req.draft = "明日は晴れかな"
    _answer(ip, partial="ええと")
    prompt = _prompt(a)
    assert "明日は晴れかな" in prompt and "ええと" in prompt


def test_what_was_seen_is_a_draft_not_the_pointer():
    a, ip = _ip()
    ip._req.lookups = [
        Lookup(
            index=1,
            action="see",
            query="部屋",
            generation=0,
            result="（見えたものは、作業状態の『観察』の行にある）",
        )
    ]
    ip._req.seen_texts = {1: "机と椅子"}
    _answer(ip)
    prompt = _prompt(a)
    assert "机と椅子" in prompt and "作業状態の" not in prompt


def test_an_unread_retry_gives_its_first_answer_and_lookup_words_are_not_drafts():
    """まだ取り込んでいない返りが言い直しなら元の答えも使う。調べものと一緒の say（声にならない合いの手）は使わない。"""
    from familiar_agent.backends import ToolCall

    a, ip = _ip()
    ip._req.lookups = [
        Lookup(index=1, action="主LLM", query="主LLM1", generation=0),
        Lookup(index=2, action="主LLM", query="主LLM2", generation=0),
    ]
    retry = _decision(
        TurnResult(stop_reason="tool_use", text="", tool_calls=[]),
        retried=True,
        original_text="元の答え",
    )
    filler = _decision(
        TurnResult(
            stop_reason="tool_use",
            text="",
            tool_calls=[
                ToolCall("s", "say", {"text": "ちょっと調べるね"}),
                ToolCall("q", "search_deferred", {"query": "天気"}),
            ],
        )
    )
    ip._drained_completions += [
        Trigger(kind="決定", query="主LLM1", decision=retry),
        Trigger(kind="決定", query="主LLM2", decision=filler),
    ]
    _answer(ip)
    prompt = _prompt(a)
    assert "元の答え" in prompt
    assert "ちょっと調べるね" not in prompt


def test_unread_camera_and_spent_results_are_not_drafts():
    """**反証の側**：まだ取り込んでいないカメラの完了は見えたものと重なり、使い切った印は結果ではない。"""
    from familiar_agent.core.completion_kind import SEARCH_SPENT

    a, ip = _ip()
    ip._req.lookups = [
        Lookup(index=1, action="see", query="部屋", generation=0),
        Lookup(index=2, action="search_deferred", query="天気", generation=0),
        Lookup(index=3, action="search_deferred", query="気温", generation=0, result=SEARCH_SPENT),
    ]
    ip._req.seen_texts = {1: "机と椅子"}
    ip._drained_completions += [
        Trigger(kind="完了", query="部屋", result="見えたもの：机と椅子", index=1),
        Trigger(kind="完了", query="天気", result=SEARCH_SPENT, index=2),
    ]
    _answer(ip)
    prompt = _prompt(a)
    assert prompt.count("机と椅子") == 1
    assert SEARCH_SPENT not in prompt


def test_stopped_lookups_are_recorded_as_aborted():
    from familiar_agent.store import intents

    a, ip = _ip()
    flying = Lookup(index=1, action="search_deferred", query="天気", generation=0)
    done = Lookup(index=2, action="search_deferred", query="気温", generation=0, result="20℃")
    ip._req.lookups = [flying, done]
    calls: list = []
    ip._ledger.close_request = lambda lks, *, state: calls.append(([lk.index for lk in lks], state))
    _answer(ip)
    assert calls[0] == ([1], intents.ABORTED)


def test_an_abort_while_writing_stops_the_answer():
    """**反証の側**：書いているあいだに打ち切られたら（名前で呼ばれた・止めるボタン）、言わず、閉じ直さない。"""
    a, ip = _ip()

    async def complete(prompt, max_tokens=300, *, system=None):
        ip._request_generation += 1  # 打ち切りが世代を進める
        ip._req.request_id = None
        return ANSWER

    a._utility_backend.complete = AsyncMock(side_effect=complete)
    assert _answer(ip) == ""
    ip._dif.speak.assert_not_awaited()
    assert _finals(a) == []


def test_a_request_that_cannot_speak_does_not_write():
    """声にならない求め（聞く相手が居ない情動）では、軽量LLM を呼ばず、録音も鳴らさずに閉じる。"""
    a, ip = _ip()
    ip._req.trigger_kind = "情動"
    ip._req.fired_axis = "bond"
    a._occupancy.return_value = 0.0  # 誰も居ない
    _answer(ip)
    a._utility_backend.complete.assert_not_awaited()
    ip._dif.say_recorded.assert_not_awaited()
    assert _finals(a)[0].endswith("時間切れになり、何も言わなかった")


def test_a_held_named_utterance_waits_for_the_time_out():
    """時間切れの知らせは、待っている名前つきの言葉より先（環-aw 決定ホ・第 1 段 項目 9。以前は言葉が先だった）。"""
    a, ip = _ip()
    words = Trigger(kind="会話入力", query="止めて", named=True)
    ip._held.append(words)
    t = Trigger(kind=el.TIMED_OUT_KIND, query="origin-1")

    async def go():
        ip._triggers.put_nowait(t)
        return await ip._take_trigger()

    assert asyncio.run(go()) is t
    assert ip._held == [words]


def test_the_time_out_goes_before_a_named_utterance():
    """同時に届いた名前つきの言葉も待たせ、時間切れの答えを先に言う（環-aw 決定ホ・第 1 段 項目 9）。"""
    a, ip = _ip()
    words = Trigger(kind="会話入力", query="止めて", named=True)
    t = Trigger(kind=el.TIMED_OUT_KIND, query="origin-1")

    async def go():
        ip._triggers.put_nowait(t)
        ip._triggers.put_nowait(words)
        return await ip._take_trigger()

    assert asyncio.run(go()) is t
    assert ip._held == [words]


def test_a_timed_out_call_does_not_recover_a_half_written_tool_call():
    """**反証の側**：途中の文が道具の呼び出しに見えても、拾い直さず時間切れのまま返す。"""
    a, ip = _ip()
    a.config.main_llm_timeout_sec = 0.05

    async def hang(**kw):
        kw["on_text"]('say(text="晴れ")')
        await asyncio.sleep(10)

    a.backend.stream_turn = hang
    result = asyncio.run(ip._main_llm_turn("sys", [], [], 100, None))
    assert result.stop_reason == el.MAIN_LLM_TIMED_OUT and not result.tool_calls


def test_the_recording_waits_for_the_cue_and_lowers_the_music(tmp_path):
    from familiar_agent.io import dif as dif_mod

    d = dif_mod.DIF.__new__(dif_mod.DIF)
    order: list = []

    async def cue():
        order.append("合図")

    async def duck(play):
        order.append("絞る")
        return await play()

    async def play_wav(path, gain):
        order.append("録音")
        return True

    (tmp_path / "timeout_answer.wav").write_bytes(b"x")

    async def go():
        d._cue_task = asyncio.ensure_future(cue())
        d._music_ducker = duck
        return await d.say_recorded("timeout_answer", gain=1.0)

    with (
        patch.object(dif_mod, "_SOUNDS", tmp_path),
        patch.object(dif_mod, "_play_wav", play_wav),
        patch("familiar_agent.core.reaction_cue.voiced", lambda p: p),
    ):
        assert asyncio.run(go()) is True
    assert order == ["合図", "絞る", "録音"]


# ── 見直し 3 回目で足した（2026-11-10） ─────────────────────────────────


def test_narration_with_a_lookup_is_not_a_draft():
    """調べものと一緒の返りは、say が無くても地の文を下書きにしない。"""
    from familiar_agent.backends import ToolCall

    a, ip = _ip()
    ip._req.lookups = [Lookup(index=1, action="主LLM", query="主LLM1", generation=0)]
    d = _decision(
        TurnResult(
            stop_reason="tool_use",
            text="調べてみます",
            tool_calls=[ToolCall("q", "search_deferred", {"query": "天気"})],
        )
    )
    ip._drained_completions.append(Trigger(kind="決定", query="主LLM1", decision=d))
    _answer(ip)
    assert "調べてみます" not in _prompt(a)


def test_a_result_used_as_a_draft_is_not_recorded_as_aborted():
    """届いていて下書きに使った完了は、打ち切ったものにしない（台帳と最終版の経緯）。"""
    from familiar_agent.store import intents

    a, ip = _ip()
    arrived = Lookup(index=1, action="search_deferred", query="天気", generation=0)
    flying = Lookup(index=2, action="search_deferred", query="気温", generation=0)
    ip._req.lookups = [arrived, flying]
    ip._drained_completions.append(Trigger(kind="完了", query="天気", result="晴れ", index=1))
    calls: list = []
    ip._ledger.close_request = lambda lks, *, state: calls.append(([lk.index for lk in lks], state))
    _answer(ip)
    assert calls[0] == ([2], intents.ABORTED)
    final = _finals(a)[0]
    assert "「天気」の結果が届いた" in final and "「気温」を打ち切った" in final


def test_a_timed_out_retry_is_written_as_timed_out():
    a, ip = _ip()
    d = _decision(
        TurnResult(stop_reason=el.MAIN_LLM_TIMED_OUT, text=""),
        retried=True,
        original_text="元の答え",
    )

    async def go():
        await ip._act_on_decision(d, utterance="明日の天気は？", gen=0)
        await ip.settle_finals()
        await ip.close()

    asyncio.run(go())
    assert "時間切れになり、「元の答え」と答えた" in _finals(a)[0]


def test_the_credit_notice_still_does_not_wait_for_the_cue(tmp_path):
    """**反証の側**：クレジット切れの知らせは、いままでどおり合図を待たず、音楽も絞らない。"""
    from familiar_agent.io import dif as dif_mod

    d = dif_mod.DIF.__new__(dif_mod.DIF)
    d._cue_task = MagicMock(done=MagicMock(return_value=False))
    d._music_ducker = AsyncMock()
    (tmp_path / "credit_gemini.wav").write_bytes(b"x")
    with (
        patch.object(dif_mod, "_SOUNDS", tmp_path),
        patch.object(dif_mod, "_play_wav", AsyncMock(return_value=True)) as play,
        patch("familiar_agent.core.reaction_cue.voiced", lambda p: p),
    ):
        assert asyncio.run(d.say_credit("gemini", gain=0.5)) is True
    play.assert_awaited_once()
    d._music_ducker.assert_not_awaited()
