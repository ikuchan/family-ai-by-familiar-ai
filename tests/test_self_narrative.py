"""Tests for SelfNarrative — persistent first-person session diary."""

from __future__ import annotations

from datetime import date

from familiar_agent.self_narrative import SelfNarrative


def test_fresh_narrative_returns_empty() -> None:
    sn = SelfNarrative()
    assert sn.read_recent() == []


def test_context_for_prompt_returns_none_when_empty() -> None:
    sn = SelfNarrative()
    assert sn.context_for_prompt() is None


def test_write_then_read_returns_entry() -> None:
    sn = SelfNarrative()
    sn.write("今日はコウタと深い話ができた。")
    entries = sn.read_recent()
    assert len(entries) == 1
    assert "コウタ" in entries[0].text


def test_write_preserves_mood() -> None:
    sn = SelfNarrative()
    sn.write("感動的な夜だった。", mood="moved")
    entries = sn.read_recent()
    assert entries[0].mood == "moved"


def test_write_preserves_trigger() -> None:
    sn = SelfNarrative()
    sn.write("まだ続いてる感じがした。", trigger="salient_turn")
    entries = sn.read_recent()
    assert entries[0].trigger == "salient_turn"


def test_write_records_date() -> None:
    sn = SelfNarrative()
    sn.write("ある日のウチ。")
    entries = sn.read_recent()
    assert entries[0].date == date.today().isoformat()


def test_write_multiple_entries_appends() -> None:
    sn = SelfNarrative()
    sn.write("一日目。")
    sn.write("二日目。")
    sn.write("三日目。")
    entries = sn.read_recent(n=10)
    assert len(entries) == 3


def test_duplicate_write_same_day_is_skipped() -> None:
    sn = SelfNarrative()
    sn.write("同じ節目。")
    sn.write("同じ節目。", trigger="salient_turn")
    entries = sn.read_recent(n=10)
    assert len(entries) == 1


def test_read_respects_n_limit() -> None:
    sn = SelfNarrative()
    for i in range(5):
        sn.write(f"日目{i}。")
    entries = sn.read_recent(n=2)
    assert len(entries) == 2


def test_read_returns_most_recent_entries() -> None:
    sn = SelfNarrative()
    sn.write("古いエントリ。")
    sn.write("新しいエントリ。")
    entries = sn.read_recent(n=1)
    assert "新しい" in entries[0].text


def test_context_for_prompt_returns_string_when_entries_exist() -> None:
    sn = SelfNarrative()
    sn.write("今日のウチ。")
    result = sn.context_for_prompt()
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


def test_context_for_prompt_includes_entry_text() -> None:
    sn = SelfNarrative()
    sn.write("星を見た夜。")
    result = sn.context_for_prompt()
    assert "星を見た夜" in result


def test_context_for_prompt_includes_up_to_three_entries() -> None:
    sn = SelfNarrative()
    for i in range(5):
        sn.write(f"エントリ{i}。")
    result = sn.context_for_prompt()
    assert result is not None
    assert "エントリ4" in result
    assert "エントリ3" in result
    assert "エントリ2" in result


def test_entries_persist_across_instances() -> None:
    sn1 = SelfNarrative()
    sn1.write("最初のセッション。")

    sn2 = SelfNarrative()
    entries = sn2.read_recent()
    assert len(entries) == 1
    assert "最初のセッション" in entries[0].text


def test_missing_file_does_not_crash() -> None:
    sn = SelfNarrative()
    assert sn.read_recent() == []
    assert sn.context_for_prompt() is None
