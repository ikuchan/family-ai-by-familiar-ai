"""見回りの行き先（知-c・2026-09-17）。

`PresenceMap.stalest_pose()` は実装だけあって呼び手が無く、`look` の `pose` は LLM が文脈から
決めていた（見ていない定点は W に浮かず「見ないから印が無く、印が無いから見に行かない」・08-01）。
ウ＝材料＋既定：情動が起点の求めでは `[いま]` に「見ていない順」の 1 行を渡し、調停が `look` を
選んで `pose` を書かなければ機械で最も長く見ていない定点を入れる。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.arbiter import _SEE_OPTION, Decision
from familiar_agent.loop.event_loop import InformationProcessing
from familiar_agent.presence_map import PresenceMap, stale_order

from tests.test_event_loop import _agent

NOW = 10_000.0


def _map():
    m = PresenceMap(poses=["出入口", "窓", "テレビ"], window_sec=180.0)
    m.mark_checked("出入口", NOW - 10)
    m.mark_checked("窓", NOW - 720)
    return m


# ── 材料 ─────────────────────────────────────────────────────────────────


def test_stale_order_lists_never_seen_first_then_oldest():
    assert stale_order(_map(), NOW) == [("テレビ", None), ("窓", 720.0), ("出入口", 10.0)]


def test_the_patrol_note_names_the_order_in_minutes():
    a = _agent(stream_returns=[])
    a._presence_sensor = MagicMock()
    a._presence_sensor.stale_order = MagicMock(return_value=stale_order(_map(), NOW))
    ip = InformationProcessing(a)
    note = ip._patrol_note()
    assert note == "\n見ていない順：テレビ 未・窓 12 分・出入口 0 分"


def test_no_sensor_means_no_note():
    a = _agent(stream_returns=[])
    a._presence_sensor = None
    assert InformationProcessing(a)._patrol_note() == ""


def test_the_arbiter_option_points_to_the_stalest_pose():
    assert "見ていない順" in _SEE_OPTION


# ── 既定 ─────────────────────────────────────────────────────────────────


def _ip_with_map(origin: str):
    a = _agent(stream_returns=[])
    a._presence_sensor = MagicMock()
    a._presence_sensor.stale_order = MagicMock(return_value=stale_order(_map(), NOW))
    a._presence_sensor.stalest_pose = MagicMock(return_value="テレビ")
    ip = InformationProcessing(a)
    ip._req = MagicMock()
    ip._req.trigger_kind = origin
    return ip


def test_an_affect_look_without_a_target_goes_to_the_stalest_pose():
    ip = _ip_with_map("情動")
    assert ip._fill_look_target({}) == {"pose": "テレビ"}


def test_a_chosen_target_is_left_alone():
    ip = _ip_with_map("情動")
    assert ip._fill_look_target({"pose": "窓"}) == {"pose": "窓"}
    assert ip._fill_look_target({"direction": "右"}) == {"direction": "右"}


def test_a_look_asked_by_a_person_is_not_filled_by_the_machine():
    ip = _ip_with_map("発話")
    assert ip._fill_look_target({}) == {}


def test_the_dispatch_uses_the_filled_target():
    ip = _ip_with_map("情動")
    ip._say_filler = AsyncMock()
    ip._start_lookup = MagicMock()
    decision = Decision(branch="action", action="look", query="首を向ける", text="", tool_input={})
    asyncio.run(ip._dispatch_arbiter_action(decision, utterance=""))
    assert ip._start_lookup.call_args.args[1]["pose"] == "テレビ"
