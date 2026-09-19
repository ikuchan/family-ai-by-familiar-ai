"""鳴る音の山谷（2026-09-19）：30 秒鳴らし、8 秒で倍率 0.1 に、25 秒を過ぎたら元の 1.5 に（本人が聴いて決めた）。

音は 1 秒の wav の繰り返し（`DIF._ring`）。繰り返すたびに、その時点の倍率（`core/ring_rules.gain_at`）で再生する。
小さい区間の倍率は比率でなく値（`RING_SOFT_GAIN`）。タイマーもアラームも同じ。
"""

from __future__ import annotations

import asyncio

from familiar_agent.config import AgentConfig
from familiar_agent.core.ring_rules import gain_at
from familiar_agent.io import dif as dif_mod
from familiar_agent.io.dif import DIF


def test_gain_is_base_then_soft_then_base_again():
    kw = dict(soft_after=8.0, soft_until=25.0, soft_gain=0.1)
    assert gain_at(0.0, 1.5, **kw) == 1.5
    assert gain_at(7.9, 1.5, **kw) == 1.5
    assert gain_at(8.0, 1.5, **kw) == 0.1
    assert gain_at(24.9, 1.5, **kw) == 0.1
    assert gain_at(25.0, 1.5, **kw) == 1.5
    assert gain_at(29.0, 1.5, **kw) == 1.5


def test_defaults_are_thirty_eight_twentyfive_tenth(monkeypatch):
    for k in (
        "TIMER_RING_SEC",
        "ALARM_RING_SEC",
        "RING_SOFT_AFTER_SEC",
        "RING_SOFT_UNTIL_SEC",
        "RING_SOFT_GAIN",
    ):
        monkeypatch.delenv(k, raising=False)
    cfg = AgentConfig()
    assert cfg.timer_ring_sec == 30.0 and cfg.alarm_ring_sec == 30.0
    assert (
        cfg.ring_soft_after_sec == 8.0
        and cfg.ring_soft_until_sec == 25.0
        and cfg.ring_soft_gain == 0.1
    )


def test_the_ring_plays_each_second_at_that_seconds_gain(monkeypatch):
    played: list[float] = []
    clock = {"t": 1000.0}

    async def fake_play(path, gain):
        played.append(gain)
        clock["t"] += 1.0  # 1 秒の wav
        return True

    monkeypatch.setattr(dif_mod, "_play_wav", fake_play)
    monkeypatch.setattr(dif_mod.time, "monotonic", lambda: clock["t"])
    d = DIF()

    async def run_ring(**kw):
        d.ring(**kw)  # create_task はループの中で
        await d._ring_task

    asyncio.run(run_ring(seconds=30.0, gain=1.5, soft_after=8.0, soft_until=25.0, soft_gain=0.1))
    assert len(played) == 30
    assert played[:8] == [1.5] * 8 and played[8:25] == [0.1] * 17 and played[25:] == [1.5] * 5
    d.configure_ring(soft_after=5.0, soft_until=10.0, soft_gain=0.2)  # T が Config から入れる形
    played.clear()
    asyncio.run(run_ring(seconds=12.0, gain=1.0))
    assert played == [1.0] * 5 + [0.2] * 5 + [1.0] * 2
