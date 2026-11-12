"""考えているあいだは声を聞かない（環-aw 第 1 段・`設計方針_パジュの状態` v0.5 決定ト、モ）。

始まりは出-dd（2026-10-31〜11-01）。10/31 21:51 に、待ちの電子音が 16 回鳴るあいだに、パジュ宛てでないとみられる言葉を
2 つ拾い、調べものの求めに添え、答えの後で新しい求めにした。いまは電子音の仕事ではなく、状態図の耳で決める：耳が
「考えているあいだ聞かない」にいるあいだは、声を名前があっても・音楽の言葉でも捨てる。

- 答えより前の考えているあいだも、確認の問いへの答え、鳴っている最中の名前つきの言葉、キーボードは捨てない（決定ト）。
- 答えを言っているあいだも耳は聞かない（決定モ）。言い終えて書き留めているあいだは聞く。
- 判定が例外を出したら聞くほうへ倒す。

電子音を鳴らせなければ仕事を終える。道具と主LLM の思わぬ例外でも完了を届ける（届かないと飛行中のまま残る）。
駆動体が例外を受けたら電子音を止める。
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.loop import event_loop as el
from familiar_agent.loop.event_loop import Trigger
from tests.test_beeps_while_waiting import _ip

CALL = "パジュ、いま何時？"
LOG = "familiar_agent.loop.event_loop"


def _gate(ip, *, query=CALL, source="voice", events=("request_opened",), voice=None) -> bool:
    """状態図へ `events` を送ってから、同じ loop で入口の門に通す（状態図は loop をまたがない）。"""
    trig = Trigger(kind="会話入力", query=query, source=source, voice=voice)

    async def go():
        for e in events:
            await ip._to_state(e)
        return await ip._swallow_if_unheard(trig)

    return asyncio.run(go())


def _quiet_ip():
    ip, _beeps = _ip()
    ip._agent._nudge_drive = AsyncMock()
    ip._agent._someone_here = MagicMock(return_value=True)
    return ip


def _ring(ip) -> None:
    """タイマーの音が鳴っている形にする。DIF はループが持つ（本体の `agent` には無い・notes_watch の記録）。"""
    ip._dif._ring_task = MagicMock(done=lambda: False)


def _dropped(caplog) -> bool:
    return any("考えているあいだの声なので捨てた" in r.getMessage() for r in caplog.records)


# ── 聞かない判定 ───────────────────────────────────────────────


def test_a_named_voice_while_thinking_is_dropped(caplog):
    ip = _quiet_ip()
    with caplog.at_level(logging.INFO, logger=LOG):
        assert _gate(ip) is True
    assert _dropped(caplog)
    assert not ip._wake_window().is_open(el.time.monotonic())  # 窓も開けない


def test_while_idle_the_same_voice_is_not_dropped(caplog):
    """反証の側：何もしていない（耳が聞いている）なら、この段では捨てない。"""
    ip = _quiet_ip()
    with caplog.at_level(logging.INFO, logger=LOG):
        _gate(ip, events=())
    assert not _dropped(caplog)


def test_a_voice_while_answering_is_dropped(caplog):
    """答えを言っているあいだも聞かない（決定モ）。"""
    ip = _quiet_ip()
    with caplog.at_level(logging.INFO, logger=LOG):
        assert _gate(ip, events=("request_opened", "speech_began")) is True
    assert _dropped(caplog)


def test_a_voice_after_the_answer_is_not_dropped(caplog):
    """反証の側：言い終えて書き留めているあいだは聞く（決定モ：列に積み、閉じてから答える）。"""
    ip = _quiet_ip()
    with caplog.at_level(logging.INFO, logger=LOG):
        _gate(ip, events=("request_opened", "speech_began", "speech_ended"))
    assert not _dropped(caplog)


def test_keyboard_while_thinking_is_taken(caplog):
    ip = _quiet_ip()
    with caplog.at_level(logging.INFO, logger=LOG):
        _gate(ip, source="keyboard")
    assert not _dropped(caplog)


def test_a_confirm_answer_while_thinking_is_taken(caplog):
    """確認の問いへの答えは捨てない（決定ト）。問いを言っているあいだに届いた答えも受ける（決定モ）。"""
    for events in (("request_opened",), ("request_opened", "speech_began")):
        ip = _quiet_ip()
        ip._confirming = lambda: True  # type: ignore[method-assign]
        with caplog.at_level(logging.INFO, logger=LOG):
            _gate(ip, query="はい", events=events)
        assert not _dropped(caplog), events


def test_a_named_voice_while_ringing_is_taken(caplog):
    """鳴っている最中の名前つきの言葉は捨てない（決定ト）。"""
    ip = _quiet_ip()
    _ring(ip)
    with caplog.at_level(logging.INFO, logger=LOG):
        _gate(ip, query="パジュ、止めて")
    assert not _dropped(caplog)


def test_an_unnamed_voice_while_ringing_is_dropped(caplog):
    """反証の側：鳴っていても、名前の無い声は捨てる。"""
    ip = _quiet_ip()
    _ring(ip)
    with caplog.at_level(logging.INFO, logger=LOG):
        assert _gate(ip, query="止めて") is True
    assert _dropped(caplog)


def test_a_music_word_while_thinking_is_dropped(caplog):
    ip = _quiet_ip()
    ip._agent._music_state.playing = True
    with caplog.at_level(logging.INFO, logger=LOG):
        assert _gate(ip, query="ストップ") is True
    assert _dropped(caplog)


def test_a_music_word_while_idle_is_taken():
    """反証の側：聞いているなら、鳴っているあいだの音楽の言葉は名前なしで受ける（出-cp）。"""
    ip = _quiet_ip()
    ip._agent._music_state.playing = True
    assert _gate(ip, query="ストップ", events=()) is False


def test_the_drop_log_has_the_length_but_not_the_words(caplog):
    ip = _quiet_ip()
    with caplog.at_level(logging.INFO, logger=LOG):
        _gate(ip)
    line = next(r.getMessage() for r in caplog.records if "考えているあいだの声" in r.getMessage())
    assert f"{len(CALL)} 字" in line
    assert "いま何時" not in line


def test_a_failing_check_falls_back_to_listening(caplog):
    ip = _quiet_ip()

    def boom():
        raise RuntimeError("壊れた")

    ip._paju().is_deaf = boom  # type: ignore[method-assign]
    with caplog.at_level(logging.INFO, logger=LOG):
        _gate(ip)
    assert not _dropped(caplog)
    assert any("聞くほうへ倒す" in r.getMessage() for r in caplog.records)


# ── 閉じたままにしない備え ─────────────────────────────────────────


def test_a_failing_beep_ends_the_beep_task(caplog):
    ip, _ = _ip()

    async def broken(kind):
        raise RuntimeError("鳴らせない")

    ip._dif.beep = broken  # type: ignore[method-assign]

    async def go():
        await ip._to_state("request_opened")
        ip._start_beeps()
        task = ip._beeps
        await asyncio.sleep(0.2)
        done, deaf = task.done(), ip._paju().is_deaf()
        await ip.close()
        return done, deaf

    with caplog.at_level(logging.WARNING, logger=LOG):
        # 鳴らせないなら、耳を聞く側へ戻す（決定ミ。求めを閉じるのは上限と一緒に入れる）。
        assert asyncio.run(go()) == (True, False)
    assert any("電子音を鳴らせなかった" in r.getMessage() for r in caplog.records)


def test_a_beep_the_player_could_not_play_ends_the_beep_task(caplog):
    """本物の再生の口は、鳴らせないと例外ではなく偽を返す。偽でも仕事を終え、耳を聞く側へ戻す（決定ミ）。"""
    ip, _ = _ip()

    async def refused(kind):
        return False

    ip._dif.beep = refused  # type: ignore[method-assign]

    async def go():
        await ip._to_state("request_opened")
        ip._start_beeps()
        task = ip._beeps
        await asyncio.sleep(0.2)
        done, deaf = task.done(), ip._paju().is_deaf()
        await ip.close()
        return done, deaf

    with caplog.at_level(logging.WARNING, logger=LOG):
        assert asyncio.run(go()) == (True, False)
    assert any("電子音を鳴らせなかった" in r.getMessage() for r in caplog.records)


def test_a_failed_beep_is_not_started_again_in_the_same_request():
    """鳴らせなかった求めでは、次の子に入っても電子音の仕事を立て直さない（失敗を毎回くり返さない）。"""
    ip, _ = _ip()
    calls: list = []

    async def broken(kind):
        calls.append(kind)
        raise RuntimeError("鳴らせない")

    ip._dif.beep = broken  # type: ignore[method-assign]

    async def go():
        await ip._to_state("request_opened")
        await asyncio.sleep(0.2)
        for event in ("decide_began", "wait_began", "check_began"):
            await ip._to_state(event)
            await asyncio.sleep(0.05)
        first = list(calls)
        ip._stop_wait_watch()  # 求めを閉じる道と同じく、電子音の仕事を片付ける
        await ip._to_state("request_closed")
        await ip._to_state("request_opened")
        await asyncio.sleep(0.2)
        await ip.close()
        return first, calls

    first, every = asyncio.run(go())
    assert first == ["recalling"]
    assert every == ["recalling", "recalling"]  # 反証の側：次の求めでは、また鳴らしてみる


def test_after_the_answer_the_failed_beep_is_tried_once_more():
    """答えを言い始めると仕事を片付けるので、答えのあとの書き留めているでは、もう 1 度だけ鳴らしてみる（くり返しは 2 回まで）。"""
    ip, _ = _ip()
    calls: list = []

    async def broken(kind):
        calls.append(kind)
        raise RuntimeError("鳴らせない")

    ip._dif.beep = broken  # type: ignore[method-assign]

    async def go():
        await ip._to_state("request_opened")
        await asyncio.sleep(0.2)
        ip._stop_beeps()  # 答えを言い始めた
        await ip._to_state("speech_began")
        await ip._to_state("speech_ended")  # 書き留めている
        await asyncio.sleep(0.2)
        await ip._to_state("note_began")
        await asyncio.sleep(0.1)
        await ip.close()

    asyncio.run(go())
    assert calls == ["recalling", "noting"]


def test_a_failed_task_left_by_an_unusual_close_does_not_silence_the_next_request():
    """片付けを通らずに閉じても（打ち切りの競り合いなど）、頭が何もしていないに入れば片付くので、次の求めは鳴る。"""
    ip, beeps = _ip()
    working = ip._dif.beep

    async def broken(kind):
        raise RuntimeError("鳴らせない")

    async def go():
        ip._dif.beep = broken  # type: ignore[method-assign]
        await ip._to_state("request_opened")
        await asyncio.sleep(0.1)
        await ip._to_state("request_closed")  # _stop_beeps を通らない
        ip._dif.beep = working  # type: ignore[method-assign]
        await ip._to_state("request_opened")
        await asyncio.sleep(0.2)
        await ip.close()

    asyncio.run(go())
    assert len(beeps) >= 2


def test_a_beep_failing_after_an_abort_leaves_the_next_request_alone():
    """**反証の側**：鳴らすあいだに世代が進んでいたら（打ち切られた）、次の求めの耳には触らない。"""
    ip, _ = _ip()

    async def broken(kind):
        ip._request_generation += 1
        raise RuntimeError("鳴らせない")

    ip._dif.beep = broken  # type: ignore[method-assign]

    async def go():
        await ip._to_state("request_opened")
        ip._start_beeps()
        await asyncio.sleep(0.2)
        deaf = ip._paju().is_deaf()
        await ip.close()
        return deaf

    assert asyncio.run(go()) is True


def test_a_working_beep_keeps_the_task():
    """反証の側：鳴らせているあいだ、仕事は続く。"""
    ip, beeps = _ip()

    async def go():
        await ip._to_state("request_opened")
        ip._start_beeps()
        task = ip._beeps
        await asyncio.sleep(0.2)
        done, deaf = task.done(), ip._paju().is_deaf()
        await ip.close()
        return done, deaf

    assert asyncio.run(go()) == (False, True)


def _completions(ip) -> "list[Trigger]":
    out = []
    while not ip._triggers.empty():
        out.append(ip._triggers.get_nowait())
    return out


def test_a_failing_music_tool_still_delivers_a_completion():
    ip, _ = _ip()
    ip._agent._music_tool = MagicMock(call=AsyncMock(side_effect=RuntimeError("壊れた")))
    asyncio.run(ip._run_lookup("play_music", {}, "曲", "iid", 1))
    got = _completions(ip)
    assert [(t.kind, t.failed, t.intent_id) for t in got] == [("完了", True, "iid")]
    assert "できたかは分からない" in got[0].result  # 道具が済んだ後の段で落ちることもある


def test_a_working_music_tool_delivers_its_result():
    """反証の側：例外が無ければ、道具の返りがそのまま届く。"""
    ip, _ = _ip()
    ip._agent._music_tool = MagicMock(call=AsyncMock(return_value=("かけた", True)))
    asyncio.run(ip._run_lookup("play_music", {}, "曲", "iid", 1))
    assert [(t.result, t.failed) for t in _completions(ip)] == [("かけた", False)]


def test_a_failing_widen_tool_still_delivers_a_completion():
    ip, _ = _ip()
    ip._run_widen = AsyncMock(side_effect=RuntimeError("壊れた"))  # type: ignore[method-assign]
    asyncio.run(ip._run_lookup(el._WIDEN_ACTIONS[0], {}, "前のこと", "iid", 2))
    assert [(t.kind, t.failed) for t in _completions(ip)] == [("完了", True)]


def test_a_cancelled_tool_is_not_swallowed():
    ip, _ = _ip()
    ip._agent._music_tool = MagicMock(call=AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(ip._run_lookup("play_music", {}, "曲", "iid", 1))
    assert _completions(ip) == []


def _main_llm(ip):
    return ip._run_main_llm(
        index=3,
        intent_id="iid",
        messages=[],
        system="sys",
        effort=None,
        capped=False,
        memories=[],
        w_id_map={},
        verdict_view="",
        recent_frame="",
        max_tokens=100,
        retried=False,
    )


def test_a_failure_around_the_main_llm_still_delivers_a_decision(monkeypatch):
    ip, _ = _ip()
    monkeypatch.setattr(el, "main_llm_cache_on", lambda: False)

    def boom(**_kw):
        raise RuntimeError("壊れた")

    ip._tools = boom  # type: ignore[method-assign]
    asyncio.run(_main_llm(ip))
    got = _completions(ip)
    assert [(t.kind, t.intent_id) for t in got] == [("決定", "iid")]
    assert got[0].decision.result.text == "" and not got[0].decision.result.tool_calls


def test_a_cancelled_main_llm_is_not_swallowed(monkeypatch):
    ip, _ = _ip()
    monkeypatch.setattr(el, "main_llm_cache_on", lambda: False)

    def cancel(**_kw):
        raise asyncio.CancelledError()

    ip._tools = cancel  # type: ignore[method-assign]
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_main_llm(ip))
    assert _completions(ip) == []


def test_an_exception_in_the_driver_stops_the_beeps():
    ip, _ = _ip()
    calls = {"n": 0}

    async def take():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("壊れた")
        await asyncio.sleep(10)

    ip._take_trigger = take  # type: ignore[method-assign]

    async def go():
        await ip._to_state(
            "request_opened"
        )  # 考えているあいだの仕事（何もしていないなら自分で終わってしまう）
        await asyncio.sleep(0.05)
        driver = asyncio.ensure_future(ip._drive())
        await asyncio.sleep(0.1)
        stopped = ip.__dict__.get("_beeps") is None
        driver.cancel()
        await ip.close()
        return stopped

    assert asyncio.run(go()) is True


def test_without_an_exception_the_driver_leaves_the_beeps():
    """反証の側：駆動体が待っているだけなら、電子音は鳴り続ける。"""
    ip, _ = _ip()

    async def take():
        await asyncio.sleep(10)

    ip._take_trigger = take  # type: ignore[method-assign]

    async def go():
        await ip._to_state(
            "request_opened"
        )  # 考えているあいだだけ鳴る（何もしていないなら仕事は終わる）
        driver = asyncio.ensure_future(ip._drive())
        await asyncio.sleep(0.1)
        alive = ip.__dict__.get("_beeps") is not None
        driver.cancel()
        await ip.close()
        return alive

    assert asyncio.run(go()) is True


# ── 見直しで足した（2026-11-01） ─────────────────────────────────────


def test_a_stranger_while_thinking_still_raises_safety():
    """知らない声の気配（知-ar）は、考えているあいだも残す。"""
    import numpy as np

    from tests.test_strangers_are_not_heard import STRANGER, _with_threshold

    ip, a, _s = _with_threshold()
    assert _gate(ip, voice=STRANGER) is True
    assert a._nudge_drive.await_count >= 1
    assert isinstance(STRANGER, np.ndarray)


def test_the_confirm_question_stops_the_beeps():
    """確認の問いへの答えは声で来るので、問う前に電子音を止める（止めないと答えを捨てる）。"""
    ip = _quiet_ip()
    ip._say_aloud = AsyncMock()  # type: ignore[method-assign]
    ip._delivery_block_reason = lambda: ""  # type: ignore[method-assign]
    ip._beeps = MagicMock(done=lambda: False)
    asyncio.run(ip._ask_confirm_aloud(MagicMock(text="3分でいい？")))
    assert ip.__dict__.get("_beeps") is None


def test_no_confirm_question_leaves_the_beeps():
    """反証の側：問いの文が無ければ、電子音はそのまま。"""
    ip = _quiet_ip()
    ip._delivery_block_reason = lambda: ""  # type: ignore[method-assign]
    ip._beeps = MagicMock(done=lambda: False)
    asyncio.run(ip._ask_confirm_aloud(MagicMock(text="")))
    assert ip.__dict__.get("_beeps") is not None


# ── 主LLM の時間切れ（2026-11-01・本人の決定。上限は 60 秒・環-aw 決定マ・test_thinking_limit） ─────────────────────────


def test_a_hung_main_llm_call_times_out_to_an_empty_result(caplog):
    ip, _ = _ip()
    ip._agent.config.main_llm_timeout_sec = 0.05

    async def hang(**_kw):
        await asyncio.sleep(10)

    ip._agent.backend.stream_turn = hang
    with caplog.at_level(logging.WARNING, logger=LOG):
        result = asyncio.run(ip._main_llm_turn("sys", [], [], 100, None))
    assert result.text == "" and not result.tool_calls
    assert any(
        "主LLM が" in r.getMessage() and "時間切れ" in r.getMessage() for r in caplog.records
    )


def test_a_quick_main_llm_call_is_not_cut():
    """反証の側：時間内に返れば、その返りをそのまま使う。"""
    from familiar_agent.backends.types import TurnResult

    ip, _ = _ip()
    ip._agent.config.main_llm_timeout_sec = 5.0

    async def quick(**_kw):
        return TurnResult(stop_reason="end_turn", text="はい"), None

    ip._agent.backend.stream_turn = quick
    assert asyncio.run(ip._main_llm_turn("sys", [], [], 100, None)).text == "はい"


# ── 見直し 2 回目で足した（2026-11-01） ─────────────────────────────────


def test_a_failure_after_the_reply_keeps_the_reply(monkeypatch):
    """返りがもうあれば、その後の段（ログの組み立て）で落ちても、返りは捨てない。"""
    from familiar_agent.backends.types import TurnResult

    ip, _ = _ip()
    monkeypatch.setattr(el, "main_llm_cache_on", lambda: False)

    async def quick(**_kw):
        return TurnResult(stop_reason="end_turn", text="はい"), None

    ip._agent.backend.stream_turn = quick

    def boom(_messages):
        raise RuntimeError("壊れた")

    monkeypatch.setattr(el, "_has_image", boom)
    asyncio.run(_main_llm(ip))
    got = _completions(ip)
    assert [t.kind for t in got] == ["決定"]
    assert got[0].decision.result.text == "はい"


def test_a_cancelled_cli_call_kills_its_child(monkeypatch):
    """主LLM の時間切れで取り消されたら、CLI の子プロセス（`claude -p`）を残さない。"""
    from familiar_agent.backends import cli

    killed: list[bool] = []

    class _Proc:
        returncode = None

        async def communicate(self, _data=None):
            await asyncio.sleep(10)

        def kill(self):
            killed.append(True)

    async def spawn(*_a, **_k):
        return _Proc()

    monkeypatch.setattr(cli.asyncio, "create_subprocess_exec", spawn)
    backend = cli.CLIBackend.__new__(cli.CLIBackend)
    backend._cmd = ["claude", "-p", "{}"]

    async def go():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(backend._run("やあ"), timeout=0.05)

    asyncio.run(go())
    assert killed == [True]


# ── 見直し 2 回目（2026-11-09）：電子音の仕事の終わり方 ─────────────────────────────


def test_a_failing_check_before_the_beep_also_takes_the_fallback(caplog):
    """鳴らす前の判定（在席など）が例外を出しても、鳴らせないときと同じく仕事を終え、耳を聞く側へ戻す（決定ミ）。"""
    ip, _ = _ip()

    def broken():
        raise RuntimeError("在席が読めない")

    ip._delivery_block_reason = broken  # type: ignore[method-assign]

    async def go():
        await ip._to_state("request_opened")
        task = ip._beeps
        await asyncio.sleep(0.2)
        done, failed, deaf = task.done(), task.done() and task.exception(), ip._paju().is_deaf()
        await ip.close()
        return done, failed, deaf

    with caplog.at_level(logging.WARNING, logger=LOG):
        assert asyncio.run(go()) == (True, None, False)
    assert any("電子音を鳴らせなかった" in r.getMessage() for r in caplog.records)


def test_the_beep_task_ends_when_the_head_leaves_thinking():
    """頭が考えているを出たら（何もしていないへ移ったら）、電子音の仕事は自分で終わり、次の求めで立て直せる。"""
    ip, beeps = _ip()

    async def go():
        await ip._to_state("request_opened")
        await asyncio.sleep(0.1)
        await ip._to_state("request_closed")  # 片付けを通らずに閉じた（打ち切りの競り合いなど）
        await asyncio.sleep(0.2)
        left = ip.__dict__.get("_beeps")
        n = len(beeps)
        await ip._to_state("request_opened")  # 次の求め
        await asyncio.sleep(0.2)
        again = len(beeps) > n
        await ip.close()
        return left, again

    left, again = asyncio.run(go())
    assert left is None and again


def test_the_closing_count_covers_the_whole_request(caplog):
    """答えの前と書き留めのあいだの両方を、閉じるときの 1 行に数える（2 本目の仕事で 0 から数え直さない）。"""
    ip, beeps = _ip()

    async def go():
        await ip._to_state("request_opened")
        await asyncio.sleep(0.2)
        ip._stop_beeps()  # 答えを言い始めた
        before = len(beeps)
        await ip._to_state("speech_began")
        await ip._to_state("speech_ended")  # 書き留めている：鳴らし直す
        await asyncio.sleep(0.2)
        ip._stop_wait_watch()  # 求めを閉じる
        total = len(beeps)
        await ip._to_state("request_closed")
        await ip.close()
        return before, total

    with caplog.at_level(logging.INFO, logger=LOG):
        before, total = asyncio.run(go())
    counts = [r.getMessage() for r in caplog.records if "待ちの電子音" in r.getMessage()]
    assert before > 0 and total > before
    assert counts[-1] == f"event-loop 待ちの電子音 {total} 回"
