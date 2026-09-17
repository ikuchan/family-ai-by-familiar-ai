"""「誰かがいる」と「知っている人がいる」を分ける（知-h・2026-09-13）。

配信ゲートの「聞く相手が居ない」は、顔の照合（知っている人）と直近 5 分の発話しか見ず、
在/不在の層（`PresenceSensor`・YOLO・登録不要）を見ていなかった。顔が未登録なら、目の前に
人が居ても「誰も居ない」になり、独り言が独白へ落ちた。順を「居るか → 誰か」にする。
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from familiar_agent.agent import EmbodiedAgent
from familiar_agent.loop.generator import _present_ctx


def _agent(*, occupied, present, last_human, said=None, speaker_at=None):
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
    if said is None:
        del a._last_said_at
    else:
        a._last_said_at = said
    if speaker_at is None:
        del a._speaker_set_at
    else:
        a._speaker_set_at = speaker_at
    a._social_presence_permission = lambda: EmbodiedAgent._social_presence_permission(a)
    return a


def test_a_body_seen_by_yolo_counts_even_if_the_face_is_unknown() -> None:
    a = _agent(occupied=True, present=[], last_human=None)
    assert a._social_presence_permission() == 1.0


def test_with_a_sensor_the_presence_table_does_not_decide_whether_anyone_is_there() -> None:
    """`/speaker パパ` が在席表に残り続け、カメラが 2 分「誰も居ない」でも自発が出た
    （2026-09-17 15:44 実機）。センサがある構成では在席表は「誰か」だけを言う。"""
    a = _agent(occupied=False, present=["p1"], last_human=time.time() - 5)
    assert a._social_presence_permission() == 0.0


def test_without_a_sensor_the_presence_table_still_counts() -> None:
    a = _agent(occupied=None, present=["p1"], last_human=None)
    assert a._social_presence_permission() == 1.0


def test_without_a_sensor_my_own_speech_still_counts() -> None:
    a = _agent(occupied=None, present=[], last_human=None, said=time.time() - 30)
    assert a._social_presence_permission() == 1.0
    b = _agent(occupied=None, present=[], last_human=time.time(), said=None)
    assert b._social_presence_permission() == 0.0


def test_a_heard_voice_alone_is_not_presence() -> None:
    """マイクは在席の証拠にしない（テレビ・物音・聞き違い・2026-09-17 17:03「こんにちは」に返事した）。"""
    a = _agent(occupied=False, present=[], last_human=time.time())
    assert a._social_presence_permission() == 0.0


def test_my_own_speech_keeps_presence_for_a_minute() -> None:
    """話してよかった状態（相手が居た）は、話してから 1 分続く（YOLO の見失いを跨ぐ）。"""
    a = _agent(occupied=False, present=[], last_human=None, said=time.time() - 30)
    assert a._social_presence_permission() == 1.0
    b = _agent(occupied=False, present=[], last_human=None, said=time.time() - 70)
    assert b._social_presence_permission() == 0.0


def test_typing_speaker_keeps_presence_for_a_minute() -> None:
    a = _agent(occupied=False, present=["p1"], last_human=None, speaker_at=time.time() - 30)
    assert a._social_presence_permission() == 1.0
    b = _agent(occupied=False, present=["p1"], last_human=None, speaker_at=time.time() - 70)
    assert b._social_presence_permission() == 0.0


def test_nobody_at_all_is_absent() -> None:
    a = _agent(occupied=False, present=[], last_human=time.time() - 600)
    assert a._social_presence_permission() == 0.0


def test_without_a_sensor_the_table_and_my_speech_decide() -> None:
    a = _agent(occupied=None, present=[], last_human=time.time())  # 声だけでは居ない
    assert a._social_presence_permission() == 0.0
    b = _agent(occupied=None, present=["p1"], last_human=None)
    assert b._social_presence_permission() == 1.0


def test_the_main_llm_is_told_someone_is_there_when_only_yolo_sees() -> None:
    a = _agent(occupied=True, present=[], last_human=time.time() - 600)
    ctx = _present_ctx(a)
    assert "unconfirmed" in ctx and "誰か居る" in ctx


def test_the_main_llm_is_told_nobody_when_nothing_sees() -> None:
    a = _agent(occupied=False, present=[], last_human=time.time() - 600)
    assert "誰も確認できていない" in _present_ctx(a)


# ── 自分の発話と /speaker が印を打つ（2026-09-17）──────────────────────────────


def test_speaking_stamps_my_own_speech_time() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from familiar_agent.loop.event_loop import InformationProcessing

    from tests.test_event_loop import _agent as _loop_agent

    a = _loop_agent(stream_returns=[])
    a._last_said_at = 0.0
    ip = InformationProcessing(a)
    ip._req = MagicMock()
    ip._req.said_fillers = []
    ip._delivery_block_reason = lambda: ""
    ip._dif = MagicMock()
    ip._dif.speak = AsyncMock()
    ip._emit = MagicMock()
    before = time.time()
    asyncio.run(ip._say_filler("見てみますね"))
    assert a._last_said_at >= before
    a._last_said_at = 0.0
    asyncio.run(ip._speak("こんにちは"))
    assert a._last_said_at >= before


def test_the_speaker_command_stamps_its_time() -> None:
    from familiar_agent.agent import EmbodiedAgent as Agent

    a = MagicMock(spec=Agent)
    a._persons = MagicMock()
    a._sync_pmm_speaker = MagicMock(return_value=None)
    import asyncio as _aio

    async def run():
        with patch("familiar_agent.agent.asyncio.ensure_future"):
            return Agent._handle_speaker_command(a, "/speaker パパ")

    before = time.time()
    _aio.run(run())
    assert a._speaker_set_at >= before
