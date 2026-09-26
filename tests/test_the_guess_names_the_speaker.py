"""写真の見立てが 1 人なら話者にし、居るかはカメラで決める（出-as 段 9a・2026-09-26・`設計方針_話していいかの決まり` §2.2・§3）。

- 見立てで家族の 1 人に当たり、ほかに名前の分からない人も居なければ、その人を話者にする（名乗りと同じ口）。
  2 人以上なら話者は「分からない」のまま。話者の寿命（60 秒・知-t）はいまのまま。今朝の実機では、話者が
  ずっと「分からない」で、黙る依頼もタイマーの沈黙も掛からなかった（知-ai）。
- 「自分が話してから 60 秒」「`/speaker` から 60 秒」は居るとみなしていた（知-h）。窓（ウェイクワード）に
  まとめたので外す。居るかはカメラ（在席センサー）で決める。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.agent import EmbodiedAgent
from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent

from tests.test_the_guess_from_a_photo_reaches_presence import _FAMILY as FAMILY  # noqa: E402


def _ip():
    a = _agent(stream_returns=[])
    a._family_md = FAMILY
    a._pmm.find_person_id_by_name = MagicMock(side_effect=lambda n: f"id:{n}")
    a._pmm.set_guessed_present = AsyncMock()
    a._pmm.note_unknown_present = MagicMock()
    ip = InformationProcessing(a)
    ip._set_speaker = AsyncMock()
    return a, ip


def test_one_guessed_person_becomes_the_speaker():
    a, ip = _ip()
    asyncio.run(ip._apply_seen_people([{"name": "パパ", "confidence": 0.6}]))
    ip._set_speaker.assert_awaited_once()
    assert ip._set_speaker.await_args.args[0] == "パパ"


def test_two_guessed_people_leave_the_speaker_unknown():
    a, ip = _ip()
    asyncio.run(
        ip._apply_seen_people(
            [{"name": "パパ", "confidence": 0.6}, {"name": "たいき", "confidence": 0.6}]
        )
    )
    ip._set_speaker.assert_not_awaited()


def test_one_named_and_one_unknown_is_two_people():
    a, ip = _ip()
    asyncio.run(
        ip._apply_seen_people(
            [{"name": "パパ", "confidence": 0.6}, {"name": "", "confidence": 0.6}]
        )
    )
    ip._set_speaker.assert_not_awaited()


# ── 居るかはカメラで決める ───────────────────────────────────────────────


def _agent_with_sensor(occupied: bool):
    a = EmbodiedAgent.__new__(EmbodiedAgent)
    sensor = MagicMock()
    sensor.room_occupied = MagicMock(return_value=occupied)
    a._presence_sensor = sensor
    a._pmm = MagicMock()
    return a


def test_having_just_spoken_no_longer_counts_as_someone_present():
    a = _agent_with_sensor(False)
    a._last_said_at = time.time()
    assert a._social_presence_permission() == 0.0


def test_a_fresh_speaker_command_no_longer_counts_as_someone_present():
    a = _agent_with_sensor(False)
    a._speaker_set_at = time.time()
    assert a._social_presence_permission() == 0.0


def test_the_camera_seeing_someone_counts():
    assert _agent_with_sensor(True)._social_presence_permission() == 1.0
