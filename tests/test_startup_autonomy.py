"""起動したら自律が回り始める（人の発話を待たない）。

`感情ループ全体像` は「起動源は Drive（欲求）の時間蓄積と発火」と定める。ところが実装では
I（情報処理機構）も T（自律機構）も在席センサも動体イベントも、**すべて `run()` の中**、
それも `user_input` があるときにしか立たなかった。起動しても、人が話しかけるまで何ひとつ
回っていない。

これは3つの症状の同じ根である。

- `/speaker` を最初に打つと入室イベントが立たない（T がまだ起動していない）
- 在席が「連続」にならない（`知覚在席` §3-2 は G（T 側・連続）と定める）
- 保留していた発話を配る起点（在席がゼロから立ち上がる瞬間）が永久に来ない

`start_autonomy()` を新設し、GUI と CUI の両方の入口から呼ぶ。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.agent import EmbodiedAgent


def _agent(*, event_loop=True, sensor=True):
    a = EmbodiedAgent.__new__(EmbodiedAgent)
    a.config = MagicMock()
    a.config.event_loop = event_loop
    a._info_processing = None
    a._tonic = None
    a._presence_sensor = MagicMock(start=AsyncMock()) if sensor else None
    a._motion_events = MagicMock(start=AsyncMock()) if sensor else None
    a._deferred_search = MagicMock()
    a._deferred_fetch = MagicMock()
    return a


def test_the_autonomous_side_starts_without_anyone_speaking():
    a = _agent()
    asyncio.run(a.start_autonomy())
    assert a._tonic is not None
    assert a._info_processing is not None


def test_the_presence_sensor_starts_too():
    a = _agent()
    asyncio.run(a.start_autonomy())
    a._presence_sensor.start.assert_awaited()
    a._motion_events.start.assert_awaited()


def test_starting_twice_does_not_make_a_second_tonic():
    a = _agent()
    asyncio.run(a.start_autonomy())
    first = a._tonic
    asyncio.run(a.start_autonomy())
    assert a._tonic is first


def test_nothing_starts_when_the_event_loop_is_off():
    # 旧経路（`EVENT_LOOP=0`）では GUI の描画ループが drive を進める。両方が進めると
    # 蓄積が二重になる。
    a = _agent(event_loop=False)
    asyncio.run(a.start_autonomy())
    assert a._tonic is None


def test_a_configuration_without_a_camera_still_starts():
    a = _agent(sensor=False)
    asyncio.run(a.start_autonomy())     # 例外を出さないこと
    assert a._tonic is not None
