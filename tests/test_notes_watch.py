"""パジュ宛てのメモを 1 時間に 1 回読み、変わっていたら自分から動く（知-g-ろ・2026-09-15）。

相手側の `get_notes_for_paju()`（家族ティア・`obsidian-memo`・HANDOVER §9）が返す本文を
`agent_state.paju_notes` と比べ、違えば `機器` の求め「パジュへのメモが変わった」を積む。
道具が無ければ黙って何もしない。何をするかは主LLM が決める。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop import notes_watch

NOTE1 = "【いま】2026-09-15 火曜日 15:00（JST）\n【出典】10_Wiki/参照/パジュへ.md（updated: 2026-09-15）\n\n## メモ\n\n- 木曜は早く帰る"
NOTE2 = "【いま】2026-09-15 火曜日 16:00（JST）\n【出典】10_Wiki/参照/パジュへ.md（updated: 2026-09-15）\n\n## メモ\n\n- 木曜は早く帰る\n- 金曜はたいきの試合"


def _agent(reply: str, ok: bool = True, connected: bool = True):
    a = MagicMock()
    a._dif.call_tool = AsyncMock(return_value=(reply, ok))
    a._dif.tool_defs = MagicMock(return_value=[{"name": "get_notes_for_paju"}] if connected else [])
    a._dif.device = MagicMock()
    return a


def test_the_body_is_what_is_compared_not_the_clock_header():
    assert notes_watch.body_of(NOTE1) == "- 木曜は早く帰る"
    assert notes_watch.body_of("【いま】…\n【出典】…\n\n## メモ\n\n（まだ無い）") == ""


def test_the_first_read_is_stored_without_waking_anyone():
    a = _agent(NOTE1)
    notes_watch._save_state(None)
    changed = asyncio.run(notes_watch.check_notes(a))
    assert changed is False
    a._dif.device.assert_not_called()
    assert notes_watch._load_state() == "- 木曜は早く帰る"


def test_a_changed_note_wakes_a_device_request_with_the_new_lines():
    a = _agent(NOTE2)
    notes_watch._save_state("- 木曜は早く帰る")
    changed = asyncio.run(notes_watch.check_notes(a))
    assert changed is True
    kind, content = a._dif.device.call_args.args[:2]
    assert kind == "メモ"
    assert "パジュへのメモが変わった" in content and "金曜はたいきの試合" in content
    assert notes_watch._load_state() == "- 木曜は早く帰る\n- 金曜はたいきの試合"


def test_an_unchanged_note_does_nothing():
    a = _agent(NOTE1)
    notes_watch._save_state("- 木曜は早く帰る")
    assert asyncio.run(notes_watch.check_notes(a)) is False
    a._dif.device.assert_not_called()


def test_a_missing_or_failing_tool_is_silent():
    a = _agent(NOTE1, connected=False)
    assert asyncio.run(notes_watch.check_notes(a)) is False
    a._dif.call_tool.assert_not_called()
    a = _agent("エラー", ok=False)
    assert asyncio.run(notes_watch.check_notes(a)) is False
    a._dif.device.assert_not_called()


def test_the_tonic_reads_once_an_hour():
    from familiar_agent.loop.tonic import Tonic

    assert notes_watch.INTERVAL_SEC == 3600.0
    t = Tonic.__new__(Tonic)
    t._notes_checked_at = None  # 起動直後は 1 回読む（前回値を持つため）
    assert t._notes_due(now=100.0) is True
    t._notes_checked_at = 100.0
    assert t._notes_due(now=100.0 + notes_watch.INTERVAL_SEC - 1) is False
    assert t._notes_due(now=100.0 + notes_watch.INTERVAL_SEC) is True
