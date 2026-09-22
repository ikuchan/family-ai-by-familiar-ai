"""名前の分からない在席者（出-ae-は・2026-09-22）。

実機 15:57、写真を見たあと機械は `unconfirmed` のまま「パパ、おかえりなさい！」と呼んだ。
**推し量ること自体は禁じない**（本人の決定）。推し量ったなら、それが在席にも使われてほしい。

そのとき在席表に入れられるのは**登録済みの人だけ**で、「居るが誰か分からない人」を表せな
かった（残課題 #8）。見立てが「2 人居て、一人はパパ、もう一人は分からない」と言っても、
分からないほうは落ちて、落ちたことも残らない。

**名前の無い在席者は `participants` に入れない。** ここは観測を書くときに人ごとの面
（`situated_memories`）を立てる材料で（`agent.py:810, 827`）、混ぜると誰にも対応しない
記憶空間に面ができる。
"""

from __future__ import annotations

import time

import pytest

from familiar_agent.person_memory_manager import PRESENCE_TIMEOUT_SEC, PersonMemoryManager


def _pmm() -> PersonMemoryManager:
    return PersonMemoryManager.__new__(PersonMemoryManager)


@pytest.fixture()
def pmm():
    import threading

    m = _pmm()
    m._present = {}
    m._spaces = {}
    m._speaker_id = None
    m._lock = threading.RLock()
    return m


# ── 名前の無い在席者を持てる ──────────────────────────────────────────────


def test_an_unknown_person_shows_up_in_the_status(pmm):
    pmm.note_unknown_present(1, confidence=0.5)
    rows = pmm.presence_status()
    assert len(rows) == 1
    assert rows[0]["name"] == "不明"
    assert rows[0]["person_id"] is None
    assert rows[0]["confidence"] == pytest.approx(0.5)


def test_an_unknown_person_is_not_a_participant(pmm):
    """面を立てる材料には入れない（誰にも対応しない記憶空間を作らないため）。"""
    pmm.note_unknown_present(2, confidence=0.5)
    assert pmm.get_present_ids() == []


def test_several_unknown_people_are_counted(pmm):
    pmm.note_unknown_present(3, confidence=0.4)
    assert len([r for r in pmm.presence_status() if r["person_id"] is None]) == 3


def test_asking_for_fewer_unknown_people_shrinks_the_list(pmm):
    """見立ては求めごとに言い直される。**足し続けない**——いまの人数に合わせる。"""
    pmm.note_unknown_present(3, confidence=0.4)
    pmm.note_unknown_present(1, confidence=0.4)
    assert len([r for r in pmm.presence_status() if r["person_id"] is None]) == 1


def test_zero_unknown_people_clears_them(pmm):
    pmm.note_unknown_present(2, confidence=0.4)
    pmm.note_unknown_present(0)
    assert pmm.presence_status() == []


def test_an_unknown_person_never_becomes_the_speaker(pmm):
    """話者には person_id が要る。名前の無い在席者は話者にならない。"""
    pmm.note_unknown_present(1, confidence=0.9)
    assert pmm._speaker_id is None
    assert all(r["is_speaker"] is False for r in pmm.presence_status())


# ── 寿命は既知の在席者と同じ ──────────────────────────────────────────────


def test_an_unknown_person_goes_stale(pmm):
    pmm.note_unknown_present(1, confidence=0.5)
    for p in pmm._present.values():
        p.last_signal_at = time.time() - PRESENCE_TIMEOUT_SEC - 1
    assert len(pmm.stale_present_ids()) == 1


def test_a_stale_unknown_person_can_be_dropped(pmm):
    pmm.note_unknown_present(1, confidence=0.5)
    for p in pmm._present.values():
        p.last_signal_at = time.time() - PRESENCE_TIMEOUT_SEC - 1
    for key in pmm.stale_present_ids():
        pmm.mark_absent(key)
    assert pmm.presence_status() == []


# ── 主LLM への渡し方 ──────────────────────────────────────────────────────


def _agent_with(rows):
    class _PMM:
        def presence_status(self):
            return rows

    class _Agent:
        _pmm = _PMM()

    return _Agent()


def test_the_unknown_person_is_told_as_others():
    from familiar_agent.loop.generator import _present_ctx

    rows = [
        {"person_id": "p1", "name": "パパ", "confidence": 0.70, "is_speaker": True},
        {"person_id": None, "name": "不明", "confidence": 0.50, "is_speaker": False},
    ]
    got = _present_ctx(_agent_with(rows))
    assert ':speaker "パパ"' in got
    assert "不明" in got


def test_only_unknown_people_means_the_speaker_is_unconfirmed():
    from familiar_agent.loop.generator import _present_ctx

    rows = [{"person_id": None, "name": "不明", "confidence": 0.50, "is_speaker": False}]
    got = _present_ctx(_agent_with(rows))
    assert ':speaker "unconfirmed"' in got
    assert "不明" in got


def test_the_note_survives_when_only_unknown_people_are_there():
    """名前の無い在席者だけのときも「名前で呼ばない」は要る（出-ae(1)）。

    注記は在席表が**空**の経路にしか無かったので、名前の無い在席者を持てるようにした
    とたん、在席表が空でなくなって注記が落ちる。15:57 の場面がまさにこれに当たる。
    """
    from familiar_agent.loop.generator import _present_ctx

    rows = [{"person_id": None, "name": "不明", "confidence": 0.50, "is_speaker": False}]
    got = _present_ctx(_agent_with(rows))
    assert "名前で呼ばない" in got


def test_the_note_is_not_added_when_someone_is_known():
    """誰かが分かっているなら、その人は名前で呼んでよい（本人の決定・2026-09-22）。"""
    from familiar_agent.loop.generator import _present_ctx

    rows = [
        {"person_id": "p1", "name": "パパ", "confidence": 0.70, "is_speaker": True},
        {"person_id": None, "name": "不明", "confidence": 0.50, "is_speaker": False},
    ]
    assert "名前で呼ばない" not in _present_ctx(_agent_with(rows))
