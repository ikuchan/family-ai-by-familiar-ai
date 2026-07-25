"""T（自律機構）の常駐タスク。

正本③の役割分担：**時計を持つのは T だけ**で、I（駆動体）は3キューを待つだけである。
drive の蓄積と発火判定は、いま GUI の描画ループに埋まっており（`gui.py:_tick_drives`）、
CUI には drive5 を進める経路が無い。T を1本立てて両方から使えるようにし、発火は AIF 経由で
QA へ積む。周期 $P_T$＝0.5 秒は課題5 A節の確定値。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from familiar_agent.core.drive_dynamics import DriveFiring
from familiar_agent.drive_register import AiDrivers
from familiar_agent.loop.tonic import TONIC_PERIOD_SEC, Tonic


def _ip():
    ip = MagicMock()
    ip.push_affect = MagicMock()
    return ip


def test_period_follows_the_confirmed_value():
    # 課題5 A節：T-tick 周期 P_T = 0.5 秒（確定）。蓄積式にも P_T が入るので勝手に変えない。
    assert TONIC_PERIOD_SEC == 0.5


def test_fires_push_affect_when_a_drive_fires():
    ip = _ip()
    firing = DriveFiring(seeking=True, rest=False, bond=False, safety=False, esteem=False)

    async def scenario():
        with patch("familiar_agent.loop.tonic.step_drives",
                   new=AsyncMock(return_value=(firing, AiDrivers()))):
            t = Tonic(ip, period=0.01)
            t.start()
            for _ in range(400):
                if ip.push_affect.call_count:
                    break
                await asyncio.sleep(0.005)
            await t.close()

    asyncio.run(scenario())
    assert ip.push_affect.called
    assert ip.push_affect.call_args.args[0] == "SEEKING"      # 発火した欲求名


def test_does_not_push_when_nothing_fires():
    ip = _ip()
    none = DriveFiring(seeking=False, rest=False, bond=False, safety=False, esteem=False)

    async def scenario():
        with patch("familiar_agent.loop.tonic.step_drives",
                   new=AsyncMock(return_value=(none, AiDrivers()))):
            t = Tonic(ip, period=0.01)
            t.start()
            await asyncio.sleep(0.08)
            await t.close()

    asyncio.run(scenario())
    assert not ip.push_affect.called


def test_close_stops_the_task():
    ip = _ip()
    none = DriveFiring(seeking=False, rest=False, bond=False, safety=False, esteem=False)

    async def scenario():
        with patch("familiar_agent.loop.tonic.step_drives",
                   new=AsyncMock(return_value=(none, AiDrivers()))):
            t = Tonic(ip, period=0.01)
            t.start()
            await asyncio.sleep(0.03)
            await t.close()
            return t._task

    task = asyncio.run(scenario())
    assert task is None or task.cancelled() or task.done()
