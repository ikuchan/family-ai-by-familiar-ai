"""待ちの電子音のあいだは声を聞かない（出-dd・2026-10-31〜11-01・本人の決定イ・ア・ア・5 分）。

10/31 21:51 に、待ちの電子音が 16 回鳴るあいだに、パジュ宛てでないとみられる言葉を 2 つ拾い、調べものの求めに添え、
答えの後で新しい求めにした。合図の声が鳴り始めてから答えの声が始まるまで（電子音の仕事が生きているあいだ）は、声を
名前があっても・音楽の言葉でも捨てる。キーボードは受ける。

**閉じたままにしない**（本人の注文）：電子音の仕事が終われば聞く。待っているものが 1 つも無いまま 5 分たてば聞く。
電子音を鳴らせなければ仕事を終える。道具と主LLM の思わぬ例外でも完了を届ける（届かないと飛行中のまま残る）。
駆動体が例外を受けたら電子音を止める。判定が例外を出したら聞くほうへ倒す。
"""

from __future__ import annotations

import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.loop import event_loop as el
from familiar_agent.loop.event_loop import Trigger
from familiar_agent.loop.request import Lookup
from tests.test_beeps_while_waiting import _ip

CALL = "パジュ、いま何時？"
LOG = "familiar_agent.loop.event_loop"


def _gate(ip, *, query=CALL, source="voice") -> bool:
    trig = Trigger(kind="会話入力", query=query, source=source)
    return asyncio.run(_gate_in_loop(ip, trig))


async def _gate_in_loop(ip, trig) -> bool:
    return await ip._swallow_if_unheard(trig)


def _beeping(ip) -> None:
    """電子音の仕事が生きている状態にする（鳴らさない・終わらない仕事を置く）。"""
    ip._beeps = MagicMock(done=lambda: False)
    ip._waited_at = time.monotonic()


def _quiet_ip():
    ip, _beeps = _ip()
    ip._load_silence = lambda: None  # type: ignore[method-assign]  # 黙っていての依頼は無い（DB を読まない）
    ip._agent._nudge_drive = AsyncMock()
    ip._agent._someone_here = MagicMock(return_value=True)
    return ip


def _dropped(caplog) -> bool:
    return any("待ちの電子音のあいだの声なので捨てた" in r.getMessage() for r in caplog.records)


# ── 聞かない判定 ───────────────────────────────────────────────


def test_a_named_voice_while_beeping_is_dropped(caplog):
    ip = _quiet_ip()
    _beeping(ip)
    with caplog.at_level(logging.INFO, logger=LOG):
        assert _gate(ip) is True
    assert _dropped(caplog)
    assert not ip._wake_window().is_open(time.monotonic())  # 窓も開けない


def test_without_beeps_the_same_voice_is_not_dropped_for_the_beeps(caplog):
    """反証の側：電子音が無ければ、この段では捨てない。"""
    ip = _quiet_ip()
    with caplog.at_level(logging.INFO, logger=LOG):
        _gate(ip)
    assert not _dropped(caplog)


def test_keyboard_while_beeping_is_taken(caplog):
    ip = _quiet_ip()
    _beeping(ip)
    with caplog.at_level(logging.INFO, logger=LOG):
        _gate(ip, source="keyboard")
    assert not _dropped(caplog)


def test_a_music_word_while_beeping_is_dropped(caplog):
    ip = _quiet_ip()
    ip._agent._music_state.playing = True
    _beeping(ip)
    with caplog.at_level(logging.INFO, logger=LOG):
        assert _gate(ip, query="ストップ") is True
    assert _dropped(caplog)


def test_a_music_word_without_beeps_is_taken():
    """反証の側：電子音が無ければ、鳴っているあいだの音楽の言葉は名前なしで受ける（出-cp）。"""
    ip = _quiet_ip()
    ip._agent._music_state.playing = True
    assert _gate(ip, query="ストップ") is False


def test_a_finished_or_stopped_beep_task_means_listening(caplog):
    ip = _quiet_ip()
    ip._beeps = MagicMock(done=lambda: True)  # 例外で落ちた・終わった仕事
    ip._waited_at = time.monotonic()
    assert ip._deaf_while_beeping() is False
    _beeping(ip)
    ip._stop_beeps()  # 答えの声で止めた
    assert ip._deaf_while_beeping() is False


def test_five_idle_minutes_reopen_listening_once(caplog):
    ip = _quiet_ip()
    _beeping(ip)
    ip._waited_at = time.monotonic() - el.DEAF_IDLE_LIMIT_SEC - 1
    with caplog.at_level(logging.WARNING, logger=LOG):
        assert ip._deaf_while_beeping() is False
        assert ip._deaf_while_beeping() is False
    warned = [r for r in caplog.records if "待つものが無いまま" in r.getMessage()]
    assert len(warned) == 1 and warned[0].levelname == "WARNING"


def test_a_long_lookup_keeps_listening_closed():
    """反証の側：調べものが飛行中なら、5 分を過ぎても聞かない（長い調べものは止めない）。"""
    ip = _quiet_ip()
    _beeping(ip)
    ip._waited_at = time.monotonic() - el.DEAF_IDLE_LIMIT_SEC - 1
    ip._req.lookups.append(Lookup(index=1, action="search_deferred", query="q", generation=0))
    assert ip._deaf_while_beeping() is True


def test_the_drop_log_has_the_length_but_not_the_words(caplog):
    ip = _quiet_ip()
    _beeping(ip)
    with caplog.at_level(logging.INFO, logger=LOG):
        _gate(ip)
    line = next(r.getMessage() for r in caplog.records if "待ちの電子音のあいだ" in r.getMessage())
    assert f"{len(CALL)} 字" in line
    assert "いま何時" not in line


def test_the_limit_is_five_minutes():
    assert el.DEAF_IDLE_LIMIT_SEC == 300.0  # 本人の決定（2026-11-01）


def test_a_failing_check_falls_back_to_listening(caplog):
    ip = _quiet_ip()
    _beeping(ip)

    def boom():
        raise RuntimeError("壊れた")

    ip._waiting_on = boom  # type: ignore[method-assign]
    ip._waited_at = time.monotonic() - el.DEAF_IDLE_LIMIT_SEC - 1
    with caplog.at_level(logging.INFO, logger=LOG):
        _gate(ip)
    assert not _dropped(caplog)
    assert any("聞くほうへ倒す" in r.getMessage() for r in caplog.records)


# ── 閉じたままにしない備え ─────────────────────────────────────────


def test_a_failing_beep_ends_the_beep_task(caplog):
    ip, _ = _ip()

    async def broken():
        raise RuntimeError("鳴らせない")

    ip._dif.beep = broken  # type: ignore[method-assign]

    async def go():
        ip._start_beeps()
        task = ip._beeps
        await asyncio.sleep(0.2)
        done = task.done()
        await ip.close()
        return done

    with caplog.at_level(logging.WARNING, logger=LOG):
        assert asyncio.run(go()) is True
    assert any("電子音を鳴らせなかった" in r.getMessage() for r in caplog.records)


def test_a_working_beep_keeps_the_task():
    """反証の側：鳴らせているあいだ、仕事は続く。"""
    ip, beeps = _ip()

    async def go():
        ip._start_beeps()
        task = ip._beeps
        await asyncio.sleep(0.2)
        done = task.done()
        await ip.close()
        return done

    assert asyncio.run(go()) is False


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
        ip._start_beeps()
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
        ip._start_beeps()
        driver = asyncio.ensure_future(ip._drive())
        await asyncio.sleep(0.1)
        alive = ip.__dict__.get("_beeps") is not None
        driver.cancel()
        await ip.close()
        return alive

    assert asyncio.run(go()) is True


# ── 見直しで足した（2026-11-01） ─────────────────────────────────────


def test_a_hung_cue_does_not_keep_listening_closed():
    """合図の声が終わらなくても、待つものが無いまま 5 分たてば聞く（合図の声は待つものに数えない）。"""
    ip = _quiet_ip()
    _beeping(ip)
    ip._cue_sound = MagicMock(done=lambda: False)
    ip._waited_at = time.monotonic() - el.DEAF_IDLE_LIMIT_SEC - 1
    assert ip._deaf_while_beeping() is False


def test_a_second_lapse_is_logged_again(caplog):
    ip = _quiet_ip()
    _beeping(ip)
    ip._waited_at = time.monotonic() - el.DEAF_IDLE_LIMIT_SEC - 1
    lk = Lookup(index=1, action="search_deferred", query="q", generation=0)
    with caplog.at_level(logging.WARNING, logger=LOG):
        ip._deaf_while_beeping()
        ip._req.lookups.append(lk)  # 待つものがまた立つ
        assert ip._deaf_while_beeping() is True
        ip._req.lookups.remove(lk)
        ip._waited_at = time.monotonic() - el.DEAF_IDLE_LIMIT_SEC - 1
        ip._deaf_while_beeping()
    assert len([r for r in caplog.records if "待つものが無いまま" in r.getMessage()]) == 2


def test_a_stranger_while_beeping_still_raises_safety():
    """知らない声の気配（知-ar）は、電子音のあいだも残す。"""
    import numpy as np

    from tests.test_strangers_are_not_heard import STRANGER, _with_threshold

    ip, a, _s = _with_threshold()
    _beeping(ip)
    trig = Trigger(kind="会話入力", query=CALL, source="voice", voice=STRANGER)
    assert asyncio.run(_gate_in_loop(ip, trig)) is True
    assert a._nudge_drive.await_count >= 1
    assert isinstance(STRANGER, np.ndarray)


def test_the_confirm_question_stops_the_beeps():
    """確認の問いへの答えは声で来るので、問う前に電子音を止める（止めないと答えを捨てる）。"""
    ip = _quiet_ip()
    ip._say_aloud = AsyncMock()  # type: ignore[method-assign]
    ip._delivery_block_reason = lambda: ""  # type: ignore[method-assign]
    _beeping(ip)
    asyncio.run(ip._ask_confirm_aloud(MagicMock(text="3分でいい？")))
    assert ip.__dict__.get("_beeps") is None
    assert ip._deaf_while_beeping() is False


def test_no_confirm_question_leaves_the_beeps():
    """反証の側：問いの文が無ければ、電子音はそのまま。"""
    ip = _quiet_ip()
    ip._delivery_block_reason = lambda: ""  # type: ignore[method-assign]
    _beeping(ip)
    asyncio.run(ip._ask_confirm_aloud(MagicMock(text="")))
    assert ip._deaf_while_beeping() is True


# ── 主LLM の時間切れ（2026-11-01・本人の決定 10 分） ─────────────────────────


def test_the_main_llm_limit_is_ten_minutes():
    assert (
        el.MAIN_LLM_TIMEOUT_SEC == 600.0
    )  # 本人の決定（2026-11-01・音楽の道具と同じく 1 回ごとに切る）


def test_a_hung_main_llm_call_times_out_to_an_empty_result(monkeypatch, caplog):
    ip, _ = _ip()
    monkeypatch.setattr(el, "MAIN_LLM_TIMEOUT_SEC", 0.05)

    async def hang(**_kw):
        await asyncio.sleep(10)

    ip._agent.backend.stream_turn = hang
    with caplog.at_level(logging.WARNING, logger=LOG):
        result = asyncio.run(ip._main_llm_turn("sys", [], [], 100, None))
    assert result.text == "" and not result.tool_calls
    assert any(
        "主LLM が" in r.getMessage() and "時間切れ" in r.getMessage() for r in caplog.records
    )


def test_a_quick_main_llm_call_is_not_cut(monkeypatch):
    """反証の側：時間内に返れば、その返りをそのまま使う。"""
    from familiar_agent.backends.types import TurnResult

    ip, _ = _ip()
    monkeypatch.setattr(el, "MAIN_LLM_TIMEOUT_SEC", 5.0)

    async def quick(**_kw):
        return TurnResult(stop_reason="end_turn", text="はい"), None

    ip._agent.backend.stream_turn = quick
    assert asyncio.run(ip._main_llm_turn("sys", [], [], 100, None)).text == "はい"


# ── 見直し 2 回目で足した（2026-11-01） ─────────────────────────────────


def test_waiting_again_in_the_beep_loop_rearms_the_lapse_log():
    """電子音の仕事が、待つものを見たら上限のログの印を戻す（声が来なくても、次の上限を残す）。"""
    ip, _ = _ip()
    ip._req.lookups.append(Lookup(index=1, action="search_deferred", query="q", generation=0))

    async def go():
        ip._start_beeps()
        ip._deaf_lapsed = True
        await asyncio.sleep(0.12)
        lapsed = ip._deaf_lapsed
        await ip.close()
        return lapsed

    assert asyncio.run(go()) is False


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
