"""#10 致命的エラー方針：DB 接続は2回リトライ、失敗なら明確な致命エラー。埋め込みは致命。"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from familiar_agent.errors import FatalStartupError


def test_connect_with_retry_succeeds_first_try():
    from familiar_agent import db

    with patch.object(db.psycopg2, "connect", return_value="CONN") as m:
        got = db._connect_with_retry("url", attempts=3, delay=0.0)
    assert got == "CONN"
    assert m.call_count == 1


def test_connect_with_retry_recovers_after_failures():
    from familiar_agent import db

    calls = {"n": 0}

    def _flaky(url):
        calls["n"] += 1
        if calls["n"] < 3:
            raise db.psycopg2.OperationalError("refused")
        return "CONN"

    with patch.object(db.psycopg2, "connect", side_effect=_flaky):
        got = db._connect_with_retry("url", attempts=3, delay=0.0)
    assert got == "CONN"
    assert calls["n"] == 3  # 2回リトライで3回目に成功


def test_connect_with_retry_raises_fatal_after_exhaustion():
    from familiar_agent import db

    with patch.object(db.psycopg2, "connect", side_effect=db.psycopg2.OperationalError("refused")):
        with pytest.raises(FatalStartupError) as ei:
            db._connect_with_retry("postgresql://u:pw@h/db", attempts=3, delay=0.0)
    msg = str(ei.value)
    assert "PostgreSQL" in msg  # 何が起きたか
    assert "pw" not in msg  # パスワードは伏せる


# ── 環-ax：起動し直した直後にネットがつながらなくても起きる、起きられなければ起こし直される ──


def test_a_cached_model_is_loaded_from_its_local_place(monkeypatch):
    """手元に控えがあれば、その場所を渡す（名前を渡すと transformers が外へ問い合わせる・11/08 21:56）。"""
    import huggingface_hub

    from familiar_agent.core import model_resource as mr

    seen: list = []

    def fake(name, **kw):
        seen.append(kw)
        return "/cache/models--x/snapshots/abc"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    assert mr.local_place("org/model") == "/cache/models--x/snapshots/abc"
    assert seen == [{"local_files_only": True}]


def test_a_model_not_cached_is_loaded_by_its_name(monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    from familiar_agent.core import model_resource as mr

    def missing(name, **kw):
        raise LocalEntryNotFoundError("手元に無い")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    assert mr.local_place("org/model") == "org/model"


def test_the_gui_exits_non_zero_after_a_fatal_startup():
    """起動の失敗のダイアログが時間切れで閉じた回は 1 で終わる。0 だと run-gui.sh は「止めた」と見て起こし直さない（11/08）。

    人が閉じた・`kill` で止めた・ふつうの起動の失敗で窓を閉じたときは 0（止める）。
    """
    from familiar_agent import gui

    class _W:
        _agent_init_failed = True
        _restart_on_exit = False

    assert gui._exit_code(_W()) == 0
    _W._restart_on_exit = True
    assert gui._exit_code(_W()) == 1
    assert gui._exit_code(object()) == 0


def test_the_fatal_dialog_closes_by_itself(monkeypatch):
    """誰もいないときに起動し直しても、ダイアログが開いたまま起こし直しを止めない。"""
    import PySide6.QtCore as qtcore
    import PySide6.QtWidgets as qtw

    from familiar_agent import gui

    timers: list = []
    shown: list = []
    closed: list = []

    class _Box:
        class Icon:
            Critical = 3

        def __init__(self, *a, **k):
            shown.append(a)

        def done(self, code):
            closed.append(code)

        def exec(self):
            return 0

    class _App:
        quit_called = False

        @staticmethod
        def instance():
            return _App

        @staticmethod
        def quit():
            _App.quit_called = True

    monkeypatch.setattr(qtw, "QMessageBox", _Box)
    monkeypatch.setattr(qtw, "QApplication", _App)
    monkeypatch.setattr(qtcore.QTimer, "singleShot", lambda ms, fn: timers.append((ms, fn)))
    w = gui.FamiliarWindow.__new__(gui.FamiliarWindow)
    monkeypatch.setattr(w, "_set_startup_status", lambda *a: None, raising=False)
    monkeypatch.setattr(w, "_set_input_enabled", lambda *a: None, raising=False)
    w._fatal_dialog("埋め込みモデルを読み込めません")
    assert shown and [ms for ms, _ in timers] == [int(gui.FATAL_DIALOG_SEC * 1000)]
    assert w._agent_init_failed is True and _App.quit_called
    assert gui._exit_code(w) == 0  # 人が先に閉じたら止める
    timers[0][1]()  # 時間切れ
    assert closed == [0] and gui._exit_code(w) == 1


def test_a_broken_local_copy_is_loaded_again_by_name(monkeypatch):
    """手元の控えが欠けていて読めなければ、名前で読み直す（外から足りないファイルを取る）。"""
    from familiar_agent.core import model_resource as mr

    monkeypatch.setattr(mr, "local_place", lambda name: "/cache/partial")
    tried: list[str] = []

    def load(p: str) -> str:
        tried.append(p)
        if p == "/cache/partial":
            raise FileNotFoundError("tokenizer.json")
        return "model"

    assert mr.load_local_first("org/model", load) == "model"
    assert tried == ["/cache/partial", "org/model"]


def test_an_unexpected_lookup_error_is_a_warning(monkeypatch, caplog):
    import logging

    import huggingface_hub

    from familiar_agent.core import model_resource as mr

    def broken(name, **kw):
        raise PermissionError("refs/main")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", broken)
    with caplog.at_level(logging.INFO, logger="familiar_agent.core.model_resource"):
        assert mr.local_place("org/model") == "org/model"
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
