"""声がしたが応じられないとき、SEEKING を押し上げる（案ア・2026-09-17）。

マイクは在席の証拠にしない（テレビ・物音・聞き違い）。しかし声がしたのに誰も見えないのは
「見に行く」理由にはなる。会話入力が入口で「誰も見えない」で止まった瞬間、SEEKING の蓄積へ
発火閾値の半分〔仮〕を足す（入口の側の test は `test_absent_hears_but_does_not_answer`）。発火は通常の tick が決める（2 回目の声で見回りへ）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.config import DriveConfig
from familiar_agent.core import drive_dynamics as dd
from familiar_agent.drive_register import AiDrivers
from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def test_nudge_adds_to_one_axis_and_stops_at_the_threshold():
    cfg = DriveConfig()
    d = AiDrivers(seeking=0.3, bond=0.2)
    out = dd.nudge(d, "seeking", 0.5, cfg)
    assert abs(out.seeking - 0.8) < 1e-9 and out.bond == 0.2
    out2 = dd.nudge(out, "seeking", 0.5, cfg)
    assert out2.seeking == cfg.theta_fire


def test_the_default_amount_is_half_the_threshold():
    cfg = DriveConfig()
    assert abs(cfg.voice_nudge - cfg.theta_fire * 0.5) < 1e-9


def _ip(origin: str, blocked: str):
    a = _agent(stream_returns=[])
    a._nudge_seeking = AsyncMock()
    ip = InformationProcessing(a)
    ip._req = MagicMock()
    ip._req.trigger_kind = origin
    ip._req.said_fillers = []
    ip._delivery_block_reason = lambda: blocked
    ip._dif = MagicMock()
    ip._dif.speak = AsyncMock()
    return a, ip


def test_a_held_reply_at_the_exit_no_longer_nudges():
    """押し上げは入口（`_swallow_if_unheard`）へ移した。出口に来る保留は機器の知らせだけ。"""
    a, ip = _ip("発話", "聞く相手が居ない")
    asyncio.run(ip._speak("こんにちは"))
    a._nudge_seeking.assert_not_awaited()


def test_a_held_device_notice_does_not_nudge():
    a, ip = _ip("機器", "聞く相手が居ない")
    asyncio.run(ip._speak("メモを読んだ"))
    a._nudge_seeking.assert_not_awaited()


def test_a_monologue_does_not_nudge():
    a, ip = _ip("情動", "聞く相手が居ない")
    asyncio.run(ip._speak("ひとりごと"))
    a._nudge_seeking.assert_not_awaited()


def test_quiet_hours_do_not_nudge():
    a, ip = _ip("発話", "静穏時間である")
    asyncio.run(ip._speak("こんにちは"))
    a._nudge_seeking.assert_not_awaited()
