"""待たせた言葉には、待たせたことと、そのあいだに鳴り終わった音を添える（環-aw 第 1 段 項目 9・2026-11-10 本人の決定ア）。

求めが開いているあいだに届いた言葉は、名前があっても待たせる（決定ホ）。答えのあとに新しい求めになったとき、主LLM には
ふつうの言葉として渡っていたので、鳴っている最中の「パジュ、止めて」が鳴り終わってから読まれると、もう止まっている音を
止めようとしかねない。届いたときの 2 つ（開いていた求め、鳴っていたか）を言葉に持たせ、W の作業状態の枠に書く。
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop import workspace
from familiar_agent.loop.event_loop import InformationProcessing, Trigger
from familiar_agent.loop.request import Request

from tests.test_event_loop import _agent

WAITED = "前の求めに答えているあいだに届き、答えのあとまで待たせた。"
RING_OVER = "届いたときに鳴っていたタイマーかアラームの音は、もう鳴り終わっている。"
RING_OVER_TAIL = "（届いたときに鳴っていたタイマーかアラームの音は、もう鳴り終わっている）"


def _w(req: Request) -> str:
    oif = MagicMock()
    oif.actors = lambda ids: {}
    oif.roles = lambda ids: {}
    text, _ = workspace.compose(oif, [], req)
    return text


# ── 届いたときのことを、言葉に持たせる ───────────────────────────────────────


def _pushed(*, open_request, ringing: bool) -> Trigger:
    a = _agent(stream_returns=[])
    a.config.agent_names = ["パジュ"]
    a._occupancy = MagicMock(return_value=1.0)
    a._nudge_drive = AsyncMock()
    ip = InformationProcessing(a)
    asyncio.run(
        ip._paju().start()
    )  # 控えを先に戻しておく（門の初めの 1 回だけ DB を読むので、短い待ちに入れない）
    ip._ensure_driver = lambda: None
    ip._timer_active = lambda: False
    ip._wake_admits = lambda *a, **k: True
    ip._req.request_id = open_request
    ip._dif = MagicMock(ringing=ringing)

    async def go():
        task = asyncio.ensure_future(ip.push_utterance("パジュ、止めて", source="keyboard"))
        await asyncio.sleep(0.01)
        got = ip._triggers.get_nowait()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return got

    return asyncio.run(go())


def test_the_word_remembers_the_ringing():
    t = _pushed(open_request="req-1", ringing=True)
    assert t.rang is True and t.waited is False  # 待たせたかは、保留箱から出たときに決まる


def test_a_word_while_quiet_remembers_nothing():
    """**反証の側**：鳴っていなければ、何も持たない。"""
    t = _pushed(open_request=None, ringing=False)
    assert t.rang is False and t.waited is False


# ── 待たせたかは、保留箱から出たかで決める ───────────────────────────────────


def test_a_word_taken_from_the_held_box_has_waited():
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    asyncio.run(
        ip._paju().start()
    )  # 控えを先に戻しておく（門の初めの 1 回だけ DB を読むので、短い待ちに入れない）
    t = Trigger(kind="会話入力", query="パジュ、止めて")
    ip._held.append(t)
    assert asyncio.run(ip._take_trigger()) is t
    assert t.waited is True


def test_a_word_taken_at_once_has_not_waited():
    """**反証の側**：求めが開いていても、待たせずにすぐ取り出した言葉は、待たせたことにしない（確認待ちの答えなど）。"""
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    asyncio.run(
        ip._paju().start()
    )  # 控えを先に戻しておく（門の初めの 1 回だけ DB を読むので、短い待ちに入れない）
    ip._req.request_id = "req-1"  # 開いているが、何も動いていない
    t = Trigger(kind="会話入力", query="はい")

    async def go():
        ip._triggers.put_nowait(t)
        return await ip._take_trigger()

    assert asyncio.run(go()) is t
    assert t.waited is False


# ── 新しい求めを開くとき ───────────────────────────────────────────────────


def _begun(trigger: Trigger, *, ringing_now: bool) -> Request:
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    asyncio.run(
        ip._paju().start()
    )  # 控えを先に戻しておく（門の初めの 1 回だけ DB を読むので、短い待ちに入れない）
    ip._dif = MagicMock(ringing=ringing_now)
    captured: dict = {}

    async def fake_iteration(utterance, voice=None):
        await ip._begin_request(kind="発話", text=utterance, utterance=utterance)
        ip._mark_ringing()
        captured["req"] = ip._req
        return ""

    ip._utterance_iteration = fake_iteration

    async def go():
        trigger.future = asyncio.get_running_loop().create_future()
        await ip._begin_utterance(trigger)

    asyncio.run(go())
    return captured["req"]


def test_a_waited_word_is_told_how_long_it_waited():
    t = Trigger(kind="会話入力", query="パジュ、止めて", waited=True)
    t.arrived = time.monotonic() - 40.4
    req = _begun(t, ringing_now=False)
    text = _w(req)
    assert "この言葉は 40 秒前、" + WAITED in text
    assert RING_OVER not in text  # 届いたときに鳴っていなかった


def test_a_waited_word_is_told_the_ringing_is_over():
    t = Trigger(kind="会話入力", query="パジュ、止めて", waited=True, rang=True)
    t.arrived = time.monotonic() - 35.0
    assert RING_OVER in _w(_begun(t, ringing_now=False))


def test_still_ringing_is_not_told_as_over():
    """**反証の側**：まだ鳴っていれば、鳴り終わったとは書かない。"""
    t = Trigger(kind="会話入力", query="パジュ、止めて", waited=True, rang=True)
    t.arrived = time.monotonic() - 5.0
    text = _w(_begun(t, ringing_now=True))
    assert WAITED in text and RING_OVER not in text


def test_a_word_that_did_not_wait_is_told_nothing():
    """**反証の側**：待たせなかった言葉には、何も書かない。"""
    t = Trigger(kind="会話入力", query="パジュ、止めて", rang=True)
    t.arrived = time.monotonic() - 40.0
    text = _w(_begun(t, ringing_now=False))
    assert WAITED not in text and RING_OVER not in text


# ── 添えた言葉 ─────────────────────────────────────────────────────────────


def _attached(*, rang: bool, ringing_now: bool) -> Request:
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    asyncio.run(
        ip._paju().start()
    )  # 控えを先に戻しておく（門の初めの 1 回だけ DB を読むので、短い待ちに入れない）
    ip._req.request_id = "req-1"
    ip._dif = MagicMock(ringing=ringing_now)

    async def go():
        fut = asyncio.get_running_loop().create_future()
        await ip._attach_to_request(
            Trigger(kind="会話入力", query="パジュ、止めて", rang=rang, future=fut)
        )
        await ip._attach_to_request(
            Trigger(
                kind="会話入力", query="あとね", future=asyncio.get_running_loop().create_future()
            )
        )
        ip._mark_ringing()

    asyncio.run(go())
    return ip._req


def test_an_added_word_is_told_the_ringing_is_over():
    text = _w(_attached(rang=True, ringing_now=False))
    assert "- 「パジュ、止めて」" + RING_OVER_TAIL in text
    assert "- 「あとね」" + RING_OVER_TAIL not in text  # 鳴っていないときに届いた言葉には付けない


def test_an_added_word_still_ringing_is_not_told():
    """**反証の側**：まだ鳴っていれば付けない。"""
    assert RING_OVER_TAIL not in _w(_attached(rang=True, ringing_now=True))
