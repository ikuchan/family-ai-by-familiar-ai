"""端-b 段 1-い：窓は、求めが開いているあいだ（考えているあいだも、答えているあいだも）閉じない（課題9 §7.4・2026-11-25 本人の決定）。

排他の仕組みは窓だけにする（窓は 1 つで、持ち主を 1 つ持つ）。いままでの窓は考えているあいだも 10 秒を数え、調べもので長く考えると
答えより前に閉じた。閉じた隙に、ほかの口の言葉が入れてしまう。保つ条件は状態図の頭から読む（考えている・答えている・声が話している）。
答えのあと 10 秒で閉じるのはいまどおり。窓が閉じているときに求めが開いても、窓は開けない。
"""

from __future__ import annotations

import asyncio
import time

from familiar_agent.loop import paju_state
from familiar_agent.loop.paju_state import PajuState

from tests.test_paju_state import _almost_lapsed, _ids


def _run(go, monkeypatch):
    monkeypatch.setattr(paju_state, "WAKE_WINDOW_SEC", 0.1)  # 閉じるまで（本物は 10 秒）

    async def run():
        sm = PajuState()
        await sm.start()
        await go(sm)

    asyncio.run(asyncio.wait_for(run(), timeout=5.0))


def test_the_window_stays_open_while_thinking_and_lapses_after_the_request(monkeypatch):
    async def go(sm):
        await sm.open_window(_almost_lapsed(0.1))
        await sm.fire("request_opened")
        await asyncio.sleep(0.4)  # 長く考える（元のままなら、もう閉じている）
        assert "open" in _ids(sm)
        await sm.fire("request_closed")
        assert "open" in _ids(sm)  # 閉じた時刻から数え直す
        await asyncio.sleep(0.4)
        assert "shut" in _ids(sm)

    _run(go, monkeypatch)


def test_the_window_lapses_after_the_answer_even_while_noting(monkeypatch):
    """答えを言い終えたら（書き留めに入ったら）、求めが開いていても 10 秒で閉じる（答えのあと 10 秒はいまどおり）。"""

    async def go(sm):
        await sm.open_window(_almost_lapsed(0.1))
        await sm.fire("request_opened")
        await sm.fire("speech_began")
        await sm.fire("voice_began")
        await asyncio.sleep(0.3)
        assert "open" in _ids(sm)  # 答えているあいだは保つ
        await sm.fire("voice_ended", played=True)
        await sm.fire("speech_ended")  # 書き留めに入る（求めはまだ開いている）
        await asyncio.sleep(0.4)
        assert "shut" in _ids(sm)

    _run(go, monkeypatch)


def test_a_quiet_request_does_not_hold_the_window(monkeypatch):
    """**反証の側**：話しかけない求め（見回り）は耳が聞いたままなので、窓を保たない。"""

    async def go(sm):
        await sm.open_window(_almost_lapsed(0.1))
        await sm.fire("quiet_request_opened")
        await asyncio.sleep(0.4)
        assert "shut" in _ids(sm)

    _run(go, monkeypatch)


def test_a_held_window_still_lapses_at_the_cap(monkeypatch):
    """求めの閉じが届かなくても（送り損ねた）、窓は上限で閉じる。"""
    monkeypatch.setattr(paju_state, "HOLD_CAP_SEC", 0.3)  # 上限の半分（保っているかの境目）も縮む

    async def go(sm):
        await sm.open_window(_almost_lapsed(0.1))
        await sm.fire("request_opened")
        await asyncio.sleep(0.15)
        assert "open" in _ids(sm)
        await asyncio.sleep(0.4)
        assert "shut" in _ids(sm)  # request_closed は来ていない

    _run(go, monkeypatch)


def test_a_request_does_not_open_a_shut_window(monkeypatch):
    """**反証の側**：窓が閉じているときに開いた求め（タイマーが鳴った知らせなど）は、窓を開けない。"""

    async def go(sm):
        await sm.fire("request_opened")
        assert "shut" in _ids(sm)
        assert not sm.window_open_at(time.monotonic())
        await sm.fire("request_closed")
        assert "shut" in _ids(sm)

    _run(go, monkeypatch)


def test_a_window_opened_while_thinking_is_held_too(monkeypatch):
    """考えているあいだに名前を呼ばれて開いた窓も、求めが閉じるまで閉じない。"""

    async def go(sm):
        await sm.fire("request_opened")
        await sm.open_window(_almost_lapsed(0.1))
        await asyncio.sleep(0.4)
        assert "open" in _ids(sm)
        await sm.fire("request_closed")
        await asyncio.sleep(0.4)
        assert "shut" in _ids(sm)

    _run(go, monkeypatch)


def test_the_window_is_held_even_if_the_ear_reopens_while_thinking(monkeypatch):
    """待ちの電子音を鳴らせないと、考え始めに耳が開く（`limit_reached`）。それでも答えるまでは窓を保つ。"""

    async def go(sm):
        await sm.open_window(_almost_lapsed(0.1))
        await sm.fire("request_opened")
        await sm.fire("limit_reached")  # 耳が聞くへ戻る
        await asyncio.sleep(0.4)
        assert "open" in _ids(sm)
        await sm.fire("request_closed")
        await asyncio.sleep(0.4)
        assert "shut" in _ids(sm)

    _run(go, monkeypatch)
