"""話しかけない発火の求めでは、耳を聞かないにしない（環-aw 決定ザ・2026-11-15・本人の決定ア）。

safety の発火は人が居ても話しかけないのに、見回しの 14〜23 秒のあいだ耳が聞かないにいて、声を 11 件捨てていた
（11/15 12:34 など）。話しかけない発火（seeking・safety）の求めは、頭は考えているへ移るが、耳は聞いているのまま。
届いた声は捨てず、この求めを打ち切って閉じ、新しい求めとして答える（§3.3・決定ジ・本人の決定イ）。
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock

import pytest

from familiar_agent.loop.event_loop import InformationProcessing, Trigger

from tests.test_event_loop import _agent


def _opened(axis: str) -> InformationProcessing:
    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._write_origin = AsyncMock(return_value=1)  # type: ignore[method-assign]
    ip._iterate = AsyncMock(return_value="")  # type: ignore[method-assign]
    ip._start_beeps = lambda: None  # type: ignore[method-assign]

    async def go():
        await ip._begin_affect(axis, "促し")
        await ip.close()  # 数える仕事を残さない（状態はそのまま）

    asyncio.run(go())
    return ip


@pytest.mark.parametrize("axis", ["safety", "seeking", "rest"])
def test_a_firing_that_never_speaks_keeps_hearing(axis):
    ip = _opened(axis)
    assert ip._paju().head_state() == "thinking"
    assert not ip._paju().is_deaf()
    assert not ip._deaf_to(Trigger(kind="会話入力", query="パジュ、いま何時？"))


@pytest.mark.parametrize("axis", ["bond", "esteem"])
def test_a_talking_firing_still_stops_hearing(axis):
    """**反証の側**：話しかける発火は、今までどおり考えているあいだ聞かない。"""
    ip = _opened(axis)
    assert ip._paju().head_state() == "thinking"
    assert ip._paju().is_deaf()


# ── 届いた声で、話しかけない求めを打ち切って答える（決定ジ・2026-11-15・本人の決定イ） ──────────────


def _voice():
    fut = asyncio.get_running_loop().create_future()
    return Trigger(kind="会話入力", query="パジュ、いま何時？", source="voice", future=fut)


def _open_with(kind: str, axis: str, action: str) -> InformationProcessing:
    from familiar_agent.loop.request import Lookup

    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._req.request_id = "req-1"
    ip._req.trigger_kind = kind
    ip._req.fired_axis = axis
    ip._req.lookups = [Lookup(index=1, action=action, query="見回す", generation=0)]
    return ip


@pytest.mark.parametrize("action", ["look", "主LLM"])
def test_words_during_a_quiet_firing_are_taken_at_once(action):
    """カメラや主LLM が飛んでいても、話しかけない求めのあいだの言葉は待たせずに取り出す。"""
    ip = _open_with("情動", "safety", action)

    async def go():
        voice = _voice()
        ip._triggers.put_nowait(voice)
        got = await asyncio.wait_for(ip._take_trigger(), timeout=1.0)
        await ip.close()
        return voice, got

    voice, got = asyncio.run(go())
    assert got is voice
    assert ip._held == []


@pytest.mark.parametrize(("kind", "axis"), [("情動", "bond"), ("機器", "")])
def test_words_during_a_talking_or_device_request_still_wait(kind, axis):
    """**反証の側**：話しかける発火と機器の求めのあいだは、いままでどおり閉じるまで待たせる（決定リ）。"""
    ip = _open_with(kind, axis, "look")

    async def go():
        ip._triggers.put_nowait(_voice())
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(ip._take_trigger(), timeout=0.1)
        await ip.close()

    asyncio.run(go())
    assert [t.query for t in ip._held] == ["パジュ、いま何時？"]


def _drive_once(ip: InformationProcessing, voice_of) -> list:
    """駆動体に言葉を 1 つ渡して回し、答え始めに渡った言葉を返す。"""
    begun: list = []

    async def go():
        voice = voice_of()
        sent = [voice]

        async def take():
            if sent:
                return sent.pop()
            await asyncio.sleep(10)

        async def begin(trigger):
            # 答え始めた時点の、前の求めの様子も残す
            begun.append(
                (trigger, ip._req.request_id, ip._paju().head_state(), ip._in_flight_count)
            )

        ip._take_trigger = take  # type: ignore[method-assign]
        ip._begin_utterance = begin  # type: ignore[method-assign]
        ip._attach_to_request = AsyncMock()  # type: ignore[method-assign]
        task = asyncio.create_task(ip._drive())
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await ip.close()

    asyncio.run(go())
    return begun


def test_the_driver_closes_the_quiet_firing_before_answering():
    """駆動体は、届いた言葉を見回しに添えず、見回しの求めを打ち切って閉じてから新しい求めとして答える。"""
    from familiar_agent.loop.request import Lookup

    ip = _opened("safety")
    ip._req.lookups = [Lookup(index=1, action="look", query="見回す", generation=0)]
    begun = _drive_once(ip, _voice)
    ip._attach_to_request.assert_not_awaited()
    assert [rest for _, *rest in begun] == [[None, "idle", 0]]


def test_a_failed_close_still_answers_the_waiting_caller():
    """打ち切りが落ちても、言葉の待ち手を待たせたままにしない（例外を渡す）。"""
    ip = _open_with("情動", "safety", "look")
    ip.abort_current = AsyncMock(side_effect=RuntimeError("閉じられない"))  # type: ignore[method-assign]
    held: list = []

    def keep():
        voice = _voice()
        held.append(voice)
        return voice

    assert _drive_once(ip, keep) == []
    assert isinstance(held[0].future.exception(), RuntimeError)


def test_closing_the_quiet_firing_tells_the_screen():
    """打ち切って閉じたことを、停止ボタンと同じく画面に知らせる（`_notify_request_state(False)`）。"""
    ip = _open_with("情動", "safety", "look")
    told: list[bool] = []
    ip._notify_request_state = told.append  # type: ignore[method-assign]
    assert len(_drive_once(ip, _voice)) == 1
    assert told == [False]


def test_a_word_nobody_waits_for_does_not_stop_the_quiet_firing():
    """**反証の側**：待ち手がもう居ない言葉（取り消された入力）では、見回しを打ち切らない。"""
    ip = _open_with("情動", "safety", "look")

    def gone():
        voice = _voice()
        voice.future.cancel()
        return voice

    _drive_once(ip, gone)
    assert ip._req.request_id == "req-1"
    assert ip._in_flight_count == 1
    ip._attach_to_request.assert_not_awaited()
