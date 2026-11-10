"""環-aw 第 1 段 項目 2：メインのループが、求めの開き閉じと、話し始め話し終わりを状態図へ送る（挙動は変えない）。

求めを開くのは `_begin_request` の 1 か所（発話・情動・機器）。声を出すのは `_say_aloud` の 1 か所（答え・申し出・確認の
問い）。閉じるのは `_finish`（いまの世代のとき）と、打ち切り（`_abort_lookups`）。名前で呼ばれても打ち切らないので（環-aw 決定ホ・
第 1 段 項目 9）、打ち切るのは停止ボタン（`abort_current`）である。
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from familiar_agent.loop.event_loop import InformationProcessing
from tests.test_event_loop import _agent


def _ip():
    a = _agent(stream_returns=[])
    a._oif.write = AsyncMock(return_value="obs-1")
    a._spawn_background_task = MagicMock()
    ip = InformationProcessing(a)
    ip._fold_told = AsyncMock()
    ip._close_exchange = MagicMock(return_value=[])
    ip._notify_request_state = MagicMock()
    return ip


def _ids(ip) -> set[str]:
    return {s.id for s in ip._paju().configuration}


@pytest.mark.parametrize("kind", ["発話", "情動", "機器"])
def test_beginning_a_request_thinks(kind):
    ip = _ip()

    async def go():
        await ip._begin_request(kind=kind, text="x", utterance="x" if kind == "発話" else "")

    asyncio.run(go())
    assert {"thinking", "deaf"} <= _ids(ip)


def test_without_a_request_it_stays_idle():
    """**反証の側**：求めを開かなければ、何もしていないのまま。"""
    ip = _ip()

    async def go():
        await ip._paju().start()

    asyncio.run(go())
    assert {"idle", "hearing"} <= _ids(ip)


def _speak_in_request(ip, *, fail: bool = False) -> set[str]:
    seen: set[str] = set()

    async def speak(text, **kw):
        seen.update(_ids(ip))
        if fail:
            raise RuntimeError("再生に失敗")

    ip._dif = MagicMock()
    ip._dif.speak = AsyncMock(side_effect=speak)

    async def go():
        await ip._begin_request(kind="発話", text="x", utterance="x")
        with pytest.raises(RuntimeError) if fail else _nothing():
            await ip._say_aloud("晴れだよ")

    asyncio.run(go())
    return seen


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_speaking_is_answering_and_after_it_notes():
    ip = _ip()
    during = _speak_in_request(ip)
    assert {"answering", "deaf"} <= during
    assert {"thinking", "noting", "hearing"} <= _ids(ip)


def test_a_failed_playback_still_ends_the_speech():
    """再生に失敗しても「話し終えた」を起こす（決定ツ）。"""
    ip = _ip()
    _speak_in_request(ip, fail=True)
    assert {"thinking", "noting", "hearing"} <= _ids(ip)


def test_finishing_the_current_request_returns_to_idle():
    ip = _ip()

    async def go():
        await ip._begin_request(kind="発話", text="x", utterance="x")
        await ip._finish("晴れだよ", [], "独白", gen=ip._request_generation)

    asyncio.run(go())
    assert {"idle", "hearing"} <= _ids(ip)


def test_finishing_an_aborted_request_does_not_close_the_next():
    """**反証の側**：打ち切られた求め（世代が進んだ）の後始末は、次の求めを閉じない。"""
    ip = _ip()

    async def go():
        await ip._begin_request(kind="発話", text="x", utterance="x")
        ip._request_generation += 1
        await ip._finish("晴れだよ", [], "独白", gen=ip._request_generation - 1)

    asyncio.run(go())
    assert "thinking" in _ids(ip)


def test_an_abort_returns_to_idle():
    ip = _ip()

    async def go():
        await ip._begin_request(kind="発話", text="x", utterance="x")
        await ip._abort_lookups()

    asyncio.run(go())
    assert {"idle", "hearing"} <= _ids(ip)


# ── 項目 3：考えているの 5 つの子へ移る事象（挙動は変えない）──────────────────────────


def _open_and_decide(ip):
    """求めを開いて、決めているまで進める（子が動くのを見分けるため、初めの子と別にしておく）。"""

    async def go():
        await ip._begin_request(kind="発話", text="x", utterance="x")
        await ip._to_state("decide_began")

    asyncio.run(go())
    assert "deciding" in _ids(ip)


def test_intake_recalls():
    """取込から思い出している（決定チ）。"""
    ip = _ip()
    _open_and_decide(ip)
    asyncio.run(ip._intake())
    assert "recalling" in _ids(ip)


def test_deciding_is_entered_by_the_decide_step():
    ip = _ip()

    async def go():
        await ip._begin_request(kind="発話", text="x", utterance="x")
        assert "recalling" in _ids(ip)
        # 主LLM が出した see の帰りは、調停を呼ばずに早く戻る道。そこでも決めているには入る。
        ip._see_returned, ip._req.see_by = True, "主LLM"
        ws = MagicMock()
        ws.window_talk = None
        await ip._decide(utterance="x", ws=ws, present_ctx="", capped=False, round_=1)

    asyncio.run(go())
    assert "deciding" in _ids(ip)


def _take_while_waiting(ip, *, in_flight: bool) -> set[str]:
    """列が空のまま `_take_trigger` を待たせ、待っているあいだの状態を返す。"""
    from familiar_agent.loop.request import Lookup

    async def go():
        await ip._begin_request(kind="発話", text="x", utterance="x")
        await ip._to_state("decide_began")
        if in_flight:
            ip._req.lookups.append(Lookup(index=1, action="主LLM", query="主LLM1", generation=0))
        task = asyncio.create_task(ip._take_trigger())
        await asyncio.sleep(0.01)
        ids = _ids(ip)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return ids

    return asyncio.run(go())


def test_waiting_for_a_return_awaits():
    """投げた仕事が飛んでいるあいだ列を待つのは、返りを待っている（決定チ）。"""
    ip = _ip()
    assert "awaiting" in _take_while_waiting(ip, in_flight=True)


def test_waiting_with_nothing_in_flight_does_not_await():
    """**反証の側**：飛んでいる仕事が無ければ、列を待っても返りを待っているにしない。"""
    ip = _ip()
    ids = _take_while_waiting(ip, in_flight=False)
    assert "deciding" in ids and "awaiting" not in ids


def _check(ip, *, on: bool) -> set[str]:
    seen: set[str] = set()

    def rows(*a, **kw):
        seen.update(_ids(ip))
        raise _Stop

    async def go():
        await ip._begin_request(kind="発話", text="x", utterance="x")
        await ip._to_state("decide_began")
        ip._agent.config.speech_check = on
        with patch("familiar_agent.loop.event_loop.past_rows", side_effect=rows):
            with contextlib.suppress(_Stop):
                await ip._speech_check_violation("晴れだよ", "", [])
        seen.update(_ids(ip))

    asyncio.run(go())
    return seen


class _Stop(Exception):
    pass


def test_the_speech_check_checks():
    ip = _ip()
    assert "checking" in _check(ip, on=True)


def test_no_speech_check_does_not_check():
    """**反証の側**：検査を切っていれば、確かめているに入らない。"""
    ip = _ip()
    ids = _check(ip, on=False)
    assert "deciding" in ids and "checking" not in ids


def test_closing_without_speech_notes():
    """声にせず閉じるときも、最終版を書くあいだは書き留めている（決定カ）。"""
    ip = _ip()
    _open_and_decide(ip)
    seen: set[str] = set()

    async def close_final(*a, **kw):
        seen.update(_ids(ip))
        return None

    ip._close_final = AsyncMock(side_effect=close_final)
    asyncio.run(ip._finish("", [], "沈黙", gen=ip._request_generation))
    assert "noting" in seen


def test_an_aborted_answer_does_not_note_the_next_request():
    """**反証の側**：打ち切られた求めの返事を書き残しても、次の求めの子は動かさない。"""
    ip = _ip()
    _open_and_decide(ip)
    ip._request_generation += 1

    asyncio.run(ip._finish("晴れだよ", [], "発話", gen=ip._request_generation - 1))
    assert "deciding" in _ids(ip)


# ── 見直し 1 回目で足した：例外で求めが閉じなくても、耳を聞く側へ戻す ──────────────────────


def test_a_failing_utterance_iteration_lets_the_ear_hear():
    """声の反復が求めを開いたあと例外で落ちても、耳は聞く側へ、頭は何もしていないへ戻る（出-dd の①を状態図で）。"""
    from familiar_agent.loop.event_loop import Trigger

    ip = _ip()

    async def broken(query, voice):
        await ip._begin_request(kind="発話", text=query, utterance=query)
        ip._beeps = MagicMock(done=lambda: False)
        raise RuntimeError("反復が落ちた")

    ip._utterance_iteration = broken

    async def go():
        fut = asyncio.get_running_loop().create_future()
        await ip._begin_utterance(Trigger(kind="会話入力", query="x", future=fut))
        assert isinstance(fut.exception(), RuntimeError)

    asyncio.run(go())
    assert {"idle", "hearing"} <= _ids(ip)
    assert ip.__dict__.get("_beeps") is None  # 電子音も止める（求めを閉じる所に届かないので）


def test_a_cancelled_utterance_lets_the_ear_hear():
    """待ち手（画面の止める）が中断して反復を止めたときも、耳を聞く側へ戻し、電子音を止める。"""
    from familiar_agent.loop.event_loop import Trigger

    ip = _ip()
    opened = asyncio.Event()

    async def slow(query, voice):
        await ip._begin_request(kind="発話", text=query, utterance=query)
        ip._beeps = MagicMock(done=lambda: False)
        opened.set()
        await asyncio.sleep(10)

    ip._utterance_iteration = slow

    async def go():
        fut = asyncio.get_running_loop().create_future()
        run = asyncio.create_task(
            ip._begin_utterance(Trigger(kind="会話入力", query="x", future=fut))
        )
        await opened.wait()
        fut.cancel()
        await run

    asyncio.run(go())
    assert {"idle", "hearing"} <= _ids(ip)
    assert ip.__dict__.get("_beeps") is None


def test_a_failing_driver_step_lets_the_ear_hear():
    from familiar_agent.loop.event_loop import Trigger

    ip = _ip()
    triggers = [Trigger(kind="情動", query="bond", result="x")]

    async def take():
        if triggers:
            return triggers.pop()
        await asyncio.sleep(10)

    async def broken(query, result):
        await ip._begin_request(kind="情動", text=result)
        ip._beeps = MagicMock(done=lambda: False)
        raise RuntimeError("求めの途中で落ちた")

    ip._take_trigger = take
    ip._swallow_if_unheard = AsyncMock(return_value=False)
    ip._begin_affect = broken

    async def go():
        task = asyncio.create_task(ip._drive())
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert {"idle", "hearing"} <= _ids(ip)
    assert ip.__dict__.get("_beeps") is None


def test_speaking_outside_a_request_is_not_logged_as_dropped(caplog):
    """求めを閉じたあとの申し出（深く調べるか）は求めの外なので、状態図に送らず、捨てたログも出さない。"""
    import logging

    ip = _ip()
    ip._dif = MagicMock()
    ip._dif.speak = AsyncMock()

    async def go():
        await ip._paju().start()
        await ip._say_aloud("もっと調べようか？")

    with caplog.at_level(logging.INFO, logger="familiar_agent.loop.paju_state"):
        asyncio.run(go())
    assert not [r for r in caplog.records if "捨てた" in r.getMessage()]
    assert {"idle", "hearing"} <= _ids(ip)


def test_an_answer_cut_off_by_an_abort_is_not_logged_as_dropped(caplog):
    """言っている途中で打ち切られたら（頭は何もしていないへ）、言い終わりを送らず、捨てたログも出さない。"""
    import logging

    ip = _ip()

    async def speak(text, **kw):
        await ip._abort_lookups()

    ip._dif = MagicMock()
    ip._dif.speak = AsyncMock(side_effect=speak)

    async def go():
        await ip._begin_request(kind="発話", text="x", utterance="x")
        await ip._say_aloud("晴れだよ")

    with caplog.at_level(logging.INFO, logger="familiar_agent.loop.paju_state"):
        asyncio.run(go())
    assert not [r for r in caplog.records if "捨てた" in r.getMessage()]
    assert {"idle", "hearing"} <= _ids(ip)


def test_a_nested_speech_does_not_end_the_outer_answer():
    """言っているあいだに別の声の口が呼ばれても（背景の確認の問い）、外の答えの途中で耳を開けない。"""
    ip = _ip()
    seen: list[set[str]] = []

    async def speak(text, **kw):
        if text == "外":
            await ip._say_aloud("内")
            seen.append(_ids(ip))

    ip._dif = MagicMock()
    ip._dif.speak = AsyncMock(side_effect=speak)

    async def go():
        await ip._begin_request(kind="発話", text="x", utterance="x")
        await ip._say_aloud("外")

    asyncio.run(go())
    assert {"answering", "deaf"} <= seen[0]
    assert {"noting", "hearing"} <= _ids(ip)


def test_waiting_with_the_head_idle_is_not_logged_as_dropped(caplog):
    """頭が何もしていないのに仕事が飛んでいる（閉じる所に届かずに戻した）ときは、返りを待っているを送らない。"""
    import logging

    from familiar_agent.loop.request import Lookup

    ip = _ip()

    async def go():
        await ip._paju().start()
        ip._req.lookups.append(Lookup(index=1, action="search_deferred", query="q", generation=0))
        task = asyncio.create_task(ip._take_trigger())
        await asyncio.sleep(0.01)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    with caplog.at_level(logging.INFO, logger="familiar_agent.loop.paju_state"):
        asyncio.run(go())
    assert not [r for r in caplog.records if "捨てた" in r.getMessage()]
