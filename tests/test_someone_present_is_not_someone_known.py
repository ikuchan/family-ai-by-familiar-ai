"""「誰かがいる」と「知っている人がいる」を分ける（知-h・2026-09-13）。

配信ゲートの「聞く相手が居ない」は、顔の照合（知っている人）と直近 5 分の発話しか見ず、
在/不在の層（`PresenceSensor`・YOLO・登録不要）を見ていなかった。顔が未登録なら、目の前に
人が居ても「誰も居ない」になり、独り言が独白へ落ちた。順を「居るか → 誰か」にする。
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from familiar_agent.agent import EmbodiedAgent
from familiar_agent.loop.generator import _present_ctx


def _agent(*, occupied, present, last_human):
    a = MagicMock()
    if occupied is None:
        a._presence_sensor = None
    else:
        a._presence_sensor = MagicMock()
        a._presence_sensor.room_occupied = MagicMock(return_value=occupied)
    a._pmm = MagicMock()
    a._pmm.get_present_ids = MagicMock(return_value=present)
    a._pmm.presence_status = MagicMock(return_value=[])
    a._persons = MagicMock()
    a._persons.active_is_explicit = False
    if last_human is None:
        del a._last_human_at
    else:
        a._last_human_at = last_human
    a._social_presence_permission = lambda: EmbodiedAgent._social_presence_permission(a)
    return a


def test_a_body_seen_by_yolo_counts_even_if_the_face_is_unknown() -> None:
    a = _agent(occupied=True, present=[], last_human=None)
    assert a._social_presence_permission() == 1.0


def test_a_known_face_still_counts() -> None:
    a = _agent(occupied=False, present=["p1"], last_human=None)
    assert a._social_presence_permission() == 1.0


def test_a_recent_voice_still_counts_without_a_body() -> None:
    a = _agent(occupied=False, present=[], last_human=time.time())
    assert a._social_presence_permission() == 1.0


def test_nobody_at_all_is_absent() -> None:
    a = _agent(occupied=False, present=[], last_human=time.time() - 600)
    assert a._social_presence_permission() == 0.0


def test_without_a_sensor_the_other_two_still_decide() -> None:
    a = _agent(occupied=None, present=[], last_human=time.time() - 600)
    assert a._social_presence_permission() == 0.0
    b = _agent(occupied=None, present=[], last_human=time.time())
    assert b._social_presence_permission() == 1.0


def test_the_main_llm_is_told_someone_is_there_when_only_yolo_sees() -> None:
    a = _agent(occupied=True, present=[], last_human=time.time() - 600)
    ctx = _present_ctx(a)
    assert "unconfirmed" in ctx and "誰か居る" in ctx


def test_the_main_llm_is_told_nobody_when_nothing_sees() -> None:
    a = _agent(occupied=False, present=[], last_human=time.time() - 600)
    assert "誰も確認できていない" in _present_ctx(a)
