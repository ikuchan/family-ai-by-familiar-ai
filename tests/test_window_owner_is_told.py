"""端-b 段 1-は：窓の持ち主が変わったら（開いた・閉じた）、つながっている端末へ知らせる（課題9 §7.4「端末での断り方」）。

- 状態図は、窓を開けた・延ばした・切れたあとに、いまの窓の持ち主（閉じていれば None）を受け手へ渡す。持ち主はデータから
  読む（切れるあいだに開け直された窓も、いまの持ち主で知らせる）。
- 端末の口は、各端末へ `{"type": "window", "busy": …}` を送る。`busy` は窓が開いていて持ち主がその端末でないとき真。
  誰と話しているかは送らない。変わらなければ送らない。つないだときは、声の設定（hello）に、いまの値を添える。
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import MagicMock

import aiohttp
import pytest

from familiar_agent.io.device_link import DeviceLink
from familiar_agent.loop.paju_state import PajuState

pytestmark = pytest.mark.real_window

TAB = "tablet-1"


def _sm():
    told: list = []
    sm = PajuState(on_window_owner=told.append)
    return sm, told


def test_opening_and_lapsing_tell_the_owner():
    async def go():
        sm, told = _sm()
        await sm.start()
        await sm.open_window(time.monotonic(), owner=TAB)
        assert told[-1] == TAB
        await sm.extend_window(time.monotonic(), owner=TAB)
        assert told[-1] == TAB
        sm.wake.until = time.monotonic() + 0.05  # すぐ切れる窓
        sm._stop_window_task()
        sm._arm_window()
        for _ in range(100):
            if told[-1] is None:
                break
            await asyncio.sleep(0.01)
        assert told[-1] is None
        sm.end_timers()

    asyncio.run(asyncio.wait_for(go(), 3.0))


def test_a_window_reopened_while_lapsing_is_told_with_its_new_owner():
    """**反証の側**：切れたことを送っているあいだに家が開け直したら、空いたとは言わず、家のものと知らせる。"""

    async def go():
        sm, told = _sm()
        await sm.start()
        await sm.open_window(time.monotonic(), owner=TAB)
        fire = sm._fire

        async def fire_and_reopen(event, *a, **k):
            if event == "window_lapsed":
                sm.wake.open(time.monotonic(), owner="")
            return await fire(event, *a, **k)

        sm._fire = fire_and_reopen
        sm.wake.until = time.monotonic() + 0.05
        sm._stop_window_task()
        sm._arm_window()
        await asyncio.sleep(0.3)
        assert told[-1] == ""
        sm.end_timers()

    asyncio.run(asyncio.wait_for(go(), 3.0))


def test_a_handover_without_a_lapse_is_told():
    """家の窓がデータの上で切れ、切れたと送る前に端末が開けても、端末のものと知らせる。"""

    async def go():
        sm, told = _sm()
        await sm.start()
        await sm.open_window(time.monotonic())
        assert told[-1] == ""
        sm.wake.until = time.monotonic() - 0.01  # 切れたが、数える仕事はまだ送っていない
        await sm.open_window(time.monotonic(), owner=TAB)
        assert told[-1] == TAB
        sm.end_timers()

    asyncio.run(asyncio.wait_for(go(), 3.0))


def test_a_stale_open_tells_the_window_is_free():
    """もう切れている古い時刻で開けても、開いたとは知らせない。"""

    async def go():
        sm, told = _sm()
        await sm.start()
        await sm.open_window(time.monotonic() - 60.0, owner=TAB)
        assert told[-1] is None
        sm.end_timers()

    asyncio.run(asyncio.wait_for(go(), 3.0))


def test_registering_the_window_listener_tells_the_current_owner():
    """口を開ける前から開いていた家の窓も、登録したときに伝わる（つないだ端末が話せると取り違えない）。"""
    a = MagicMock()
    a.config.agent_names = ["パジュ"]
    from familiar_agent.loop.event_loop import InformationProcessing

    ip = InformationProcessing(a)

    async def go():
        await ip._paju().start()
        await ip._paju().open_window(time.monotonic())
        got: list = []
        ip.set_window_listener(got.append)
        ip._paju().end_timers()
        return got

    assert asyncio.run(asyncio.wait_for(go(), 3.0)) == [""]


def test_the_loop_passes_the_owner_to_its_window_listener():
    a = MagicMock()
    a.config.agent_names = ["パジュ"]
    from familiar_agent.loop.event_loop import InformationProcessing

    ip = InformationProcessing(a)
    got: list = []
    ip.set_window_listener(got.append)

    async def go():
        await ip._paju().start()
        await ip._paju().open_window(time.monotonic(), owner=TAB)
        ip._paju().end_timers()

    asyncio.run(asyncio.wait_for(go(), 3.0))
    assert got[-1] == TAB


# ── 端末の口 ──────────────────────────────────────────────────────────────


async def _noop(device, text, speaker):
    return None


def _run(body):
    async def go():
        link = DeviceLink(_noop, {"model": "m", "voice": "v"}, host="127.0.0.1", port=0)
        assert await link.start()
        try:
            async with aiohttp.ClientSession() as session:
                return await asyncio.wait_for(
                    body(link, session, f"http://127.0.0.1:{link.port}"), 5.0
                )
        finally:
            await link.close()

    return asyncio.run(go())


async def _recv(ws) -> dict:
    return json.loads((await asyncio.wait_for(ws.receive(), 2.0)).data)


async def _connect(session, url, name, *, busy=False):
    ws = await session.ws_connect(f"{url}/device?name={name}")
    hello = await _recv(ws)
    assert hello["type"] == "hello" and hello["busy"] is busy
    return ws


def test_each_device_is_told_whether_it_may_talk():
    async def body(link, session, url):
        tab = await _connect(session, url, TAB)
        phone = await _connect(session, url, "phone-2")
        link.window(TAB)
        assert await _recv(phone) == {
            "type": "window",
            "busy": True,
        }  # 持ち主（tab）はグレーにしないので送らない
        link.window(None)
        assert await _recv(phone) == {"type": "window", "busy": False}
        link.window("")  # 家
        assert await _recv(tab) == {
            "type": "window",
            "busy": True,
        }  # tab へ届く最初の知らせ（前の 2 回は変わらなかった）
        assert await _recv(phone) == {"type": "window", "busy": True}

    _run(body)


def test_an_unchanged_owner_is_not_sent_again():
    async def body(link, session, url):
        link.window("")
        tab = await _connect(session, url, TAB, busy=True)  # つないだときに、いまの値
        link.window("")
        link.window("")
        link.window(None)
        assert await _recv(tab) == {"type": "window", "busy": False}  # 間に同じ値は送っていない

    _run(body)


def test_the_window_notice_does_not_say_who_is_talking():
    async def body(link, session, url):
        phone = await _connect(session, url, "phone-2")
        link.window(TAB)
        msg = await _recv(phone)
        assert TAB not in json.dumps(msg) and set(msg) == {"type", "busy"}

    _run(body)


def test_the_agent_registers_the_link_as_the_window_listener():
    from familiar_agent.config import DeviceLinkConfig

    from tests.test_device_link import _agent

    async def go():
        a = _agent(DeviceLinkConfig(port=0))
        await a.start_autonomy()
        link = a._device_link
        try:
            a._info_processing.set_window_listener.assert_called_with(link.window)
        finally:
            await a._close_device_link()

    asyncio.run(go())
