"""顔ぶれは人ごとの持ち時間で保つ（知-ai 段 2・2026-10-05・本人の決定・`設計方針_在席と顔ぶれ` v0.1）。

**顔ぶれ**（特定の誰がいるか）は、その人だと分かる印（写真で見つけた・声が当たった・名乗りが当たった・`/voice`）
が来るたびに、人ごとの**持ち時間** 1 分（`PRESENCE_HOLD_SEC`・仮）を数え直す。切れたら外す。

以前は上下が逆だった：在席（カメラ）が「誰も居ない」を 60 秒見続けると顔ぶれ表を全員消し、写真の見立てが来る
たびに見立ての人を置き換え、1 人なら話者にしていた。首を回して写らなくなっただけでも消え、SEEKING の見回りの
たびに顔ぶれと話者が揺れた。

- 在席が「いない」でも、持ち時間のあいだは消さない。
- 見立ては、写った人を入れるか数え直すだけ。写っていない人は消さない（持ち時間で切れる）。
- 見立てで話者は付けない（話者は声・名乗り・`/speaker`・`[名前]` が決める）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.person_memory_manager import PersonMemoryManager

HOLD = 60.0


@pytest.fixture
def pmm(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr("familiar_agent.person_memory_manager.time.time", lambda: clock["t"])
    base = MagicMock()
    base.list_persons = MagicMock(return_value=[])
    m = PersonMemoryManager(base)
    m.clock = clock  # type: ignore[attr-defined]
    return m


def test_the_hold_is_one_minute(monkeypatch):
    from familiar_agent.config import AgentConfig

    monkeypatch.delenv("PRESENCE_HOLD_SEC", raising=False)
    cfg = AgentConfig()
    assert cfg.presence_hold_sec == 60.0
    assert not hasattr(cfg, "presence_expire_sec")  # 在席で顔ぶれを消す失効は撤去


def test_a_person_stays_for_the_hold_and_then_goes(pmm):
    asyncio.run(pmm.person_arrived("papa"))
    pmm.clock["t"] += 59
    assert pmm.expire_presence(HOLD) == []
    assert pmm.get_present_ids() == ["papa"]
    pmm.clock["t"] += 2
    assert pmm.expire_presence(HOLD) == ["papa"]
    assert pmm.get_present_ids() == []


def test_a_new_sign_restarts_the_hold(pmm):
    asyncio.run(pmm.person_arrived("papa"))
    pmm.clock["t"] += 50
    asyncio.run(pmm.set_speaker("papa", source="voice"))  # 声が当たった
    pmm.clock["t"] += 50
    assert pmm.expire_presence(HOLD) == []


def test_a_guess_adds_without_removing_others(pmm):
    asyncio.run(pmm.set_guessed_present([("papa", 0.6)]))
    asyncio.run(pmm.set_guessed_present([("taiki", 0.6)]))  # 首を回して別の人だけ写った
    assert sorted(pmm.get_present_ids()) == ["papa", "taiki"]


def test_a_guess_refreshes_someone_already_there(pmm):
    asyncio.run(pmm.person_arrived("papa"))  # 声で入った
    pmm.clock["t"] += 50
    asyncio.run(pmm.set_guessed_present([("papa", 0.6)]))
    pmm.clock["t"] += 50
    assert pmm.expire_presence(HOLD) == []
    row = next(r for r in pmm._present.values() if r.person_id == "papa")
    assert row.source == ""  # 由来は最初の入り方のまま


def test_unknown_people_are_kept_until_their_hold_ends(pmm):
    pmm.note_unknown_present(2)
    pmm.note_unknown_present(0)  # 写真に写らなかった
    assert len([k for k in pmm.present_keys() if k.startswith("unknown:")]) == 2
    pmm.clock["t"] += 61
    pmm.expire_presence(HOLD)
    assert pmm.present_keys() == []


# ── T：在席では顔ぶれを消さない ─────────────────────────────────────────────


def test_tonic_expires_presence_by_its_hold_not_by_occupancy():
    from familiar_agent.loop.tonic import Tonic

    agent = MagicMock()
    agent.config.presence_hold_sec = 60.0
    agent._pmm.expire_presence = MagicMock(return_value=[])
    sensor = MagicMock()
    sensor.room_occupied = MagicMock(return_value=False)
    t = Tonic(MagicMock(), agent=agent, occupancy=sensor)
    t._expire_presence()
    agent._pmm.expire_presence.assert_called_once_with(60.0)
    agent._pmm.mark_absent.assert_not_called()


# ── 見立てでは話者を付けない ─────────────────────────────────────────────────


def test_a_single_guess_does_not_name_the_speaker():
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    a._family_md = "## パパ\n- **名前**：雄輔\n- **呼び方**：パパ\n"
    a._pmm = MagicMock()
    a._pmm.find_person_id_by_name = MagicMock(return_value="papa")
    a._pmm.set_guessed_present = AsyncMock()
    ip = InformationProcessing(a)
    ip._set_speaker = AsyncMock()  # type: ignore[method-assign]
    asyncio.run(ip._apply_seen_people([{"name": "パパ", "confidence": 0.9}]))
    a._pmm.set_guessed_present.assert_awaited_once()
    ip._set_speaker.assert_not_awaited()
