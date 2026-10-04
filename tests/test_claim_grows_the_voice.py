"""名乗りで声の基準を育てる（知-ae 段 5・2026-10-02 → 知-ai 段 4・2026-10-05 で改めた）。

名乗り（「パジュ、パパだよ」）の声が名乗った本人に当たって話者に付いたとき（`voice_claim_threshold`・0.30）、その
発話の声を、その人の**登録の声**（日をまたいで残る・上限 30）と**今日の声**（その日だけ・上限 10）に足す。
本人の声の基準がまだ無いときは名乗りでは育てない（最初の登録は `/voice`・知-ai）。名乗りの預かりは撤去した。

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


def test_a_claim_whose_voice_matches_grows_registered_and_today():
    ip, a, s = _ip(present=0.0, speaker=None, store=_Store(registered={"papa": PAPA}))
    ip._req.voice = PAPA
    _claim(ip, "パパ")
    a._persons.set_active.assert_called_once_with("パパ")
    assert s.added == [("papa", "voice", "registered", 30), ("papa", "voice", "today", 10)]


def test_a_claim_by_the_current_speaker_still_grows():
    ip, a, s = _ip(speaker="パパ", store=_Store(registered={"papa": PAPA}))
    a._persons.active_name = "パパ"
    ip._req.voice = PAPA
    _claim(ip, "パパ")
    assert [x[2] for x in s.added] == ["registered", "today"]


def test_no_voice_or_an_unknown_name_grows_nothing():
    ip, a, s = _ip(speaker=None, store=_Store(registered={"papa": PAPA}))
    ip._req.voice = None
    _claim(ip, "パパ")
    assert s.added == []
    ip._req.voice = PAPA
    _claim(ip, "太郎")
    assert s.added == []


def test_a_claim_by_someone_without_a_voice_grows_nothing():
    """最初の登録は `/voice`（知-ai・本人の決定ウ）。名乗りだけでは育てない。"""
    ip, a, s = _ip(speaker=None, store=_Store())
    ip._req.voice = PAPA
    _claim(ip, "パパ")
    a._persons.set_active.assert_not_called()
    assert s.added == []


def test_a_strict_match_alone_never_grows_registered():
    ip, a, s = _ip()
    asyncio.run(ip._match_voice(TAIKI))
    assert all(origin != "registered" for _, _, origin, _ in s.added)


def test_a_store_failure_does_not_stop_the_claim():
    ip, a, s = _ip(speaker=None, store=_Store(registered={"papa": PAPA}))
    s.add = MagicMock(side_effect=RuntimeError("db"))  # type: ignore[method-assign]
    ip._req.voice = PAPA
    _claim(ip, "パパ")
    a._persons.set_active.assert_called_once_with("パパ")
    assert isinstance(a._sync_pmm_speaker, AsyncMock)
    assert np.allclose(ip._req.voice, PAPA)
