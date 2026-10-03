"""名乗りで声の基準を育てる（知-ae 段 5・2026-10-02・本人の決定ウ・`設計方針_声で話者を見分ける` v0.1）。

録音の手順は作らない。名乗り（「パジュ、パパだよ」）が話者に付いたとき、その発話の声を、その人の**登録の声**
（日をまたいで残る・上限 30）と**今日の声**（その日だけ・上限 10）に足す。

- 声のある発話で話者がはっきり分かるのは名乗りだけ（`/speaker`・`[名前]` はキーボード）。
- 預かった名乗り（在席が無かった・知-w-ろ）は、声も一緒に預け、付いたときに足す。捨てたときは足さない。
- 家族に無い名前・声の特徴が無い発話では足さない。
- 厳しい閾値の当たり（段 4）は今日の声にだけ足し、登録の声には足さない——推定が日をまたいで残ると、取り違えが
  積み重なる。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from familiar_agent.loop import arbiter
from tests.test_voice_picks_the_speaker import PAPA, TAIKI, _ip, _Store


def _claim(ip, name: str):
    asyncio.run(
        ip._apply_speaker_claim(arbiter.Decision(branch="light", text="x", speaker_claim=name))
    )


def test_a_claim_with_presence_grows_registered_and_today():
    ip, a, s = _ip(speaker=None, store=_Store())
    ip._req.voice = PAPA
    _claim(ip, "パパ")
    a._persons.set_active.assert_called_once_with("パパ")
    assert s.added == [("papa", "voice", "registered", 30), ("papa", "voice", "today", 10)]


def test_a_claim_by_the_current_speaker_still_grows():
    ip, a, s = _ip(speaker="パパ", store=_Store())
    a._persons.active_name = "パパ"
    ip._req.voice = PAPA
    _claim(ip, "パパ")
    assert [x[2] for x in s.added] == ["registered", "today"]


def test_no_voice_or_an_unknown_name_grows_nothing():
    ip, a, s = _ip(speaker=None, store=_Store())
    ip._req.voice = None
    _claim(ip, "パパ")
    assert s.added == []
    ip._req.voice = PAPA
    _claim(ip, "太郎")
    assert s.added == []


def test_a_kept_claim_carries_its_voice_and_grows_when_it_lands(monkeypatch):
    ip, a, s = _ip(present=0.0, speaker=None, store=_Store())
    ip._req.voice = TAIKI
    monkeypatch.setattr("familiar_agent.loop.event_loop.time.time", lambda: 1000.0)
    _claim(ip, "たいき")
    assert s.added == []
    ip._req.voice = PAPA  # 次の求めは別の発話（声は預けたほうを使う）
    a._social_presence_permission = MagicMock(return_value=1.0)
    monkeypatch.setattr("familiar_agent.loop.event_loop.time.time", lambda: 1010.0)
    asyncio.run(ip._apply_pending_claim())
    a._persons.set_active.assert_called_once_with("たいき")
    assert s.added == [("taiki", "voice", "registered", 30), ("taiki", "voice", "today", 10)]


def test_an_expired_kept_claim_grows_nothing(monkeypatch):
    ip, a, s = _ip(present=0.0, speaker=None, store=_Store())
    ip._req.voice = TAIKI
    monkeypatch.setattr("familiar_agent.loop.event_loop.time.time", lambda: 1000.0)
    _claim(ip, "たいき")
    a._social_presence_permission = MagicMock(return_value=1.0)
    monkeypatch.setattr("familiar_agent.loop.event_loop.time.time", lambda: 1100.0)
    asyncio.run(ip._apply_pending_claim())
    assert s.added == []


def test_a_strict_match_alone_never_grows_registered():
    ip, a, s = _ip()
    asyncio.run(ip._judge_voice(TAIKI))
    assert all(origin != "registered" for _, _, origin, _ in s.added)


def test_a_store_failure_does_not_stop_the_claim():
    ip, a, s = _ip(speaker=None, store=_Store())
    s.add = MagicMock(side_effect=RuntimeError("db"))  # type: ignore[method-assign]
    ip._req.voice = PAPA
    _claim(ip, "パパ")
    a._persons.set_active.assert_called_once_with("パパ")
    assert isinstance(a._sync_pmm_speaker, AsyncMock)
    assert np.allclose(ip._req.voice, PAPA)
