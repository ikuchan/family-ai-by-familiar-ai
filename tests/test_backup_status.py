"""Tests for _backup_status_note() in EmbodiedAgent."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path


def _make_agent():
    from familiar_agent.agent import EmbodiedAgent

    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    return agent


def _write_log(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


class TestBackupStatusNote:
    def test_returns_empty_when_no_log_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        agent = _make_agent()
        assert agent._backup_status_note() == ""

    def test_returns_empty_when_recent_backup(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        ts = (datetime.now() - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%S")
        log = tmp_path / ".familiar_ai" / "backups" / "backup.log"
        _write_log(log, [
            f"[{ts}] Starting backup → familiar_ai_20260609.sql.gz",
            f"[{ts}] Done: 820K",
            f"[{ts}] Upload complete",
        ])
        agent = _make_agent()
        assert agent._backup_status_note() == ""

    def test_returns_warning_when_stale(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        ts = (datetime.now() - timedelta(hours=30)).strftime("%Y-%m-%dT%H:%M:%S")
        log = tmp_path / ".familiar_ai" / "backups" / "backup.log"
        _write_log(log, [
            f"[{ts}] Starting backup → familiar_ai.sql.gz",
            f"[{ts}] Done: 820K",
        ])
        agent = _make_agent()
        note = agent._backup_status_note()
        assert note != ""
        assert "30" in note or "h ago" in note

    def test_returns_warning_when_no_done_entry(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        log = tmp_path / ".familiar_ai" / "backups" / "backup.log"
        _write_log(log, [
            f"[{ts}] Starting backup → familiar_ai.sql.gz",
            "Error: something went wrong",
        ])
        agent = _make_agent()
        note = agent._backup_status_note()
        assert note != ""
        assert "no successful" in note.lower() or "record" in note.lower()

    def test_uses_most_recent_done_entry(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        old_ts = (datetime.now() - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%S")
        new_ts = (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
        log = tmp_path / ".familiar_ai" / "backups" / "backup.log"
        _write_log(log, [
            f"[{old_ts}] Done: 800K",
            f"[{new_ts}] Done: 820K",
        ])
        agent = _make_agent()
        # Most recent is 2h ago → should be silent
        assert agent._backup_status_note() == ""
