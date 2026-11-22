"""#10 致命的エラー方針：DB 接続は2回リトライ、失敗なら明確な致命エラー。埋め込みは致命。"""

from __future__ import annotations

from types import SimpleNamespace
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


def _fatal_fakes(monkeypatch):
    """`_fatal_dialog` の Qt を偽物にする。返す箱に、開いた・閉じた・時計・後片付けが残る。"""
    import PySide6.QtCore as qtcore
    import PySide6.QtWidgets as qtw

    from familiar_agent import gui

    seen = SimpleNamespace(
        boxes=[], timers=[], shown=[], closed=[], opened=[], modality=[], quit=0, shutdown=0
    )

    class _Box:
        class Icon:
            Critical = 3

        def __init__(self, *a, **k):
            seen.shown.append(a)
            seen.boxes.append(self)
            self._on_finished: list = []
            self.finished = SimpleNamespace(connect=self._on_finished.append)
            self._visible = False

        def setWindowModality(self, m):  # noqa: N802
            seen.modality.append(m)

        def isVisible(self):  # noqa: N802
            return self._visible

        def done(self, code):
            seen.closed.append(code)
            self._visible = False
            for fn in self._on_finished:
                fn(code)

        def setObjectName(self, name):  # noqa: N802
            self.name = name

        def show(self):
            seen.opened.append(True)
            self._visible = True

        def open(self):  # open() はモーダルを窓だけに戻すので使わない
            raise AssertionError("open で開いた")

        # 入れ子の輪は使わない（11/14：開いているあいだ、ほかの仕事に入れず 6,068 回の例外）
        def exec(self):
            raise AssertionError("exec で開いた")

    class _App:
        @staticmethod
        def instance():
            return _App

        @staticmethod
        def quit():
            seen.quit += 1

    monkeypatch.setattr(qtw, "QMessageBox", _Box)
    monkeypatch.setattr(qtw, "QApplication", _App)
    monkeypatch.setattr(qtcore.QTimer, "singleShot", lambda ms, fn: seen.timers.append((ms, fn)))
    w = gui.FamiliarWindow.__new__(gui.FamiliarWindow)
    monkeypatch.setattr(w, "_set_startup_status", lambda *a: None, raising=False)
    monkeypatch.setattr(w, "_set_input_enabled", lambda *a: None, raising=False)

    w._shutdown_done = False
    monkeypatch.setattr(w, "close", lambda: setattr(seen, "quit", seen.quit + 1), raising=False)

    def _shutdown_now():
        seen.shutdown += 1

    monkeypatch.setattr(w, "_ensure_shutdown_task", _shutdown_now, raising=False)
    return gui, w, seen


def test_the_fatal_dialog_closes_by_itself(monkeypatch):
    """誰もいないときに起動し直しても、ダイアログが開いたまま起こし直しを止めない。"""
    import PySide6.QtCore as qtcore

    gui, w, seen = _fatal_fakes(monkeypatch)
    w._fatal_dialog("埋め込みモデルを読み込めません")
    assert (
        seen.shown
        and seen.opened
        and [ms for ms, _ in seen.timers] == [int(gui.FATAL_DIALOG_SEC * 1000)]
    )
    assert seen.modality == [
        qtcore.Qt.WindowModality.ApplicationModal
    ]  # exec と同じく、ほかの窓も止める
    assert (
        seen.shutdown == 1
    )  # 開く前に後片付けを始め、自律と書き起こしを止める（壊れた記憶で話さない）
    assert w._agent_init_failed is True and seen.quit == 0  # 開いただけでは終わらない
    seen.timers[0][1]()  # 時間切れ（窓を閉じる。後片付けが済んでいなければ、済んでから閉じる）
    assert seen.closed == [0] and seen.quit == 1 and gui._exit_code(w) == 1


def test_a_fatal_dialog_closed_by_a_person_stops_for_good(monkeypatch):
    """人が先に閉じたら 0 で終わり、あとから時計が来ても、起こし直しにしない。"""
    gui, w, seen = _fatal_fakes(monkeypatch)
    w._fatal_dialog("埋め込みモデルを読み込めません")
    seen.boxes[0].done(0)  # 人が閉じた
    assert seen.quit == 1 and gui._exit_code(w) == 0
    seen.timers[0][1]()  # 60 秒の時計があとから来た
    assert gui._exit_code(w) == 0 and seen.closed == [0]


def test_a_failed_cleanup_still_counts_as_done():
    """後片付けの途中で失敗しても済んだことにする（済まないと、窓が閉じられないまま残る）。"""
    import asyncio

    from familiar_agent import gui

    w = gui.FamiliarWindow.__new__(gui.FamiliarWindow)
    w._shutdown_done = False

    async def broken():
        raise RuntimeError("見張りの仕事が落ちていた")

    w._shutdown_steps = broken
    with pytest.raises(RuntimeError):
        asyncio.run(w._shutdown())
    assert w._shutdown_done is True


def test_the_window_waits_only_for_the_fatal_dialog(monkeypatch):
    """後片付けが先に済んでも、起動の失敗のダイアログが開いているあいだは窓を閉じない。ほかのダイアログでは待たない。"""
    from familiar_agent import gui

    w = gui.FamiliarWindow.__new__(gui.FamiliarWindow)
    closed: list = []
    monkeypatch.setattr(w, "close", lambda: closed.append(True), raising=False)
    w._shutdown_done = True
    box = SimpleNamespace(isVisible=lambda: True)
    asked: list = []

    def find(cls, name):
        asked.append(name)
        return box

    monkeypatch.setattr(w, "findChild", find, raising=False)
    w._finalize_close()
    assert closed == [] and asked == [gui._FATAL_BOX]
    monkeypatch.setattr(
        w, "findChild", lambda cls, name: None, raising=False
    )  # 開いていない（ほかのダイアログだけ）
    w._finalize_close()
    assert closed == [True]


def test_the_cleanup_runs_only_once(monkeypatch):
    """済んだあとの `aboutToQuit` で、後片付けをもう一度始めない（閉じた agent をもう一度閉じない）。"""
    from familiar_agent import gui

    w = gui.FamiliarWindow.__new__(gui.FamiliarWindow)
    done = SimpleNamespace(done=lambda: True)
    w._shutdown_task = done
    made: list = []
    monkeypatch.setattr(w, "_create_task", lambda coro: made.append(coro), raising=False)
    assert w._ensure_shutdown_task() is done and made == []


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
