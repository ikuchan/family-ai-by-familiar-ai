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
    """ループ（`InformationProcessing`）の偽物。DIF はループが持つ（agent には無い）。"""
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent as _real_agent

    ip = InformationProcessing(_real_agent(stream_returns=[]))
    ip._dif = MagicMock()
    ip._dif.call_tool = AsyncMock(return_value=(reply, ok))
    ip._dif.tool_defs = MagicMock(
        return_value=[{"name": "get_notes_for_paju"}] if connected else []
    )
    ip._dif.device = MagicMock()
    ip.record_device = AsyncMock()  # メモは記憶に記録だけする（出-as §2.7）
    return ip


def test_the_body_is_what_is_compared_not_the_clock_header():
    assert notes_watch.body_of(NOTE1) == "- 木曜は早く帰る"
    assert notes_watch.body_of("【いま】…\n【出典】…\n\n## メモ\n\n（まだ無い）") == ""


def test_the_first_read_is_stored_without_waking_anyone():
    a = _agent(NOTE1)
    notes_watch._save_state(None)
    changed = asyncio.run(notes_watch.check_notes(a))
    assert changed is False
    a.record_device.assert_not_awaited()
    assert notes_watch._load_state() == "- 木曜は早く帰る"


def test_a_changed_note_is_recorded_with_the_new_lines():
    a = _agent(NOTE2)
    notes_watch._save_state("- 木曜は早く帰る")
    changed = asyncio.run(notes_watch.check_notes(a))
    assert changed is True
    kind, content = a.record_device.await_args.args[:2]
    assert kind == "メモ"
    assert "パジュへのメモが変わった" in content and "金曜はたいきの試合" in content
    assert notes_watch._load_state() == "- 木曜は早く帰る\n- 金曜はたいきの試合"


def test_an_unchanged_note_does_nothing():
    a = _agent(NOTE1)
    notes_watch._save_state("- 木曜は早く帰る")
    assert asyncio.run(notes_watch.check_notes(a)) is False
    a.record_device.assert_not_awaited()


def test_a_missing_or_failing_tool_is_silent():
    a = _agent(NOTE1, connected=False)
    assert asyncio.run(notes_watch.check_notes(a)) is False
    a._dif.call_tool.assert_not_called()
    a = _agent("エラー", ok=False)
    assert asyncio.run(notes_watch.check_notes(a)) is False
    a.record_device.assert_not_awaited()


def test_the_tonic_reads_once_an_hour():
    from familiar_agent.loop.tonic import Tonic

    assert notes_watch.INTERVAL_SEC == 3600.0
    # T はループ（`_ip`）を渡す。agent には DIF が無い（実機で落ちた・2026-09-15）。
    import inspect

    src = inspect.getsource(Tonic._maybe_check_notes)
    assert "check_notes(self._ip)" in src
    t = Tonic.__new__(Tonic)
    t._notes_checked_at = None  # 起動直後は 1 回読む（前回値を持つため）
    assert t._notes_due(now=100.0) is True
    t._notes_checked_at = 100.0
    assert t._notes_due(now=100.0 + notes_watch.INTERVAL_SEC - 1) is False
    assert t._notes_due(now=100.0 + notes_watch.INTERVAL_SEC) is True


def test_the_record_is_written_before_the_state_is_saved():
    """記録で落ちたら前回値は進まない（差分を失わない・実機 19:40）。"""
    a = _agent(NOTE2)
    notes_watch._save_state("- 木曜は早く帰る")
    a.record_device = AsyncMock(side_effect=RuntimeError("記録できない"))
    try:
        asyncio.run(notes_watch.check_notes(a))
    except RuntimeError:
        pass
    assert notes_watch._load_state() == "- 木曜は早く帰る"


def test_the_loops_dif_can_push_a_device_event():
    """ループの DIF は `ip` を持つ（`device()` が落ちない）。"""
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent as _real_agent

    ip = InformationProcessing(_real_agent(stream_returns=[]))
    ip.push_device = MagicMock()
    ip._dif.device("メモ", "x")
    ip.push_device.assert_called_once()
