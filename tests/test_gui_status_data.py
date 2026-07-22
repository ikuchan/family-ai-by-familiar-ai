"""GUI 状態表示のデータ層：drives 読み取りと、話者/在席の統合ビュー。

GUI 描画は PyQt 依存で単体テストしにくいので、状態→表示データの変換（PMM の
アクセサ・drives の引数なし読み取り）だけをここで検証する。描画は実機確認。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_load_current_drives_roundtrip():
    """引数なしで現在の drive5 を読める（load_current_mood と同型）。"""
    from familiar_agent.db import get_db
    from familiar_agent.drive_register import AiDrivers, load_current_drives, save_drives

    db = get_db()
    with db.lock:
        conn = db.conn()
        save_drives(conn, AiDrivers(seeking=0.7, bond=0.3))
        conn.commit()

    d = load_current_drives()
    assert isinstance(d, AiDrivers)
    assert d.seeking == pytest.approx(0.7)
    assert d.bond == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_set_speaker_retains_source_and_confidence():
    """set_speaker が source と confidence を保持し、speaker_status が返す。"""
    from familiar_agent.person_memory_manager import PersonMemoryManager, RecognitionHint

    base = MagicMock()
    base.list_persons.return_value = [
        {"id": "p1", "name": "alice", "display_name": "アリス"}
    ]
    pmm = PersonMemoryManager(base, switch_thresholds={"face": 0.4})
    await pmm.apply_hint(RecognitionHint(person_id="p1", confidence=0.66, source="face"))

    st = pmm.speaker_status()
    assert st is not None
    assert st["name"] == "アリス"
    assert st["source"] == "face"
    assert st["confidence"] == pytest.approx(0.66)


def test_speaker_status_none_when_no_speaker():
    from familiar_agent.person_memory_manager import PersonMemoryManager

    pmm = PersonMemoryManager(MagicMock())
    assert pmm.speaker_status() is None


@pytest.mark.asyncio
async def test_presence_status_lists_present_with_speaker_flag():
    """presence_status が在席者の name・confidence・is_speaker を返す。"""
    from familiar_agent.person_memory_manager import PersonMemoryManager

    base = MagicMock()
    base.list_persons.return_value = [
        {"id": "p1", "name": "alice", "display_name": "アリス"},
        {"id": "p2", "name": "bob", "display_name": "ボブ"},
    ]
    pmm = PersonMemoryManager(base)
    await pmm.person_arrived("p1", confidence=0.9)
    await pmm.person_arrived("p2", confidence=0.5)
    await pmm.set_speaker("p1", source="manual")

    rows = {r["name"]: r for r in pmm.presence_status()}
    assert set(rows) == {"アリス", "ボブ"}
    assert rows["アリス"]["is_speaker"] is True
    assert rows["ボブ"]["is_speaker"] is False
    assert rows["アリス"]["confidence"] == pytest.approx(0.9)
