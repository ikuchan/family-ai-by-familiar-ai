"""吹き出しを**足した**ことがログで分かる（2026-09-12 実機で露見）。

`chat.log` は「表示関数に渡した」までしか証言しない。渡したあと吹き出しを足したか、
足いたあと画面内に来たかは、ログが無いと画面を見ている人にしか分からない。
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


@pytest.fixture
def chat_log(qapp, tmp_path, monkeypatch):
    # `ChatLog.__init__` は `~/.cache/familiar-ai/chat.log` にセッション開始を書く。
    # テストで実物の chat.log を汚さない。
    monkeypatch.setattr("familiar_agent.gui.Path.home", lambda: tmp_path)
    from familiar_agent.gui import ChatLog

    log = ChatLog(agent_label="パジュ")
    log.resize(400, 300)
    log.show()
    yield log
    # 遅らせた記録が、次のテストで消えた窓を指さないように、ここで1周まわして消化する。
    qapp.processEvents()
    log.close()


def test_adding_a_bubble_leaves_a_line_in_the_log(chat_log, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="familiar_agent.gui"):
        chat_log.append_line("[パジュ] 目の前を見ますね。")
    lines = [r.getMessage() for r in caplog.records if "吹き出し" in r.getMessage()]
    assert lines, "吹き出しを足したことがログに残らない"
    # 何件目か・スクロール位置・先頭の文字が、1行で読める。
    assert "1件目" in lines[0]
    assert "目の前を見ますね。" in lines[0]
    assert "スクロール=" in lines[0]


def test_the_second_bubble_is_counted_as_the_second(chat_log, caplog) -> None:
    chat_log.append_line("[パジュ] はい、確認しますね。")
    with caplog.at_level(logging.INFO, logger="familiar_agent.gui"):
        chat_log.append_line("[パジュ] 目の前を見ますね。")
    lines = [r.getMessage() for r in caplog.records if "吹き出し" in r.getMessage()]
    assert lines and "2件目" in lines[0]


def test_after_the_event_loop_turns_the_bubble_reports_visibility(chat_log, qapp, caplog) -> None:
    """足した直後は Qt がまだ描いていない。1周まわったあとに可視かを改めて残す。"""
    with caplog.at_level(logging.INFO, logger="familiar_agent.gui"):
        chat_log.append_line("[パジュ] 目の前を見ますね。")
        qapp.processEvents()
        qapp.processEvents()
    later = [r.getMessage() for r in caplog.records if "描いたあと" in r.getMessage()]
    assert later, "描いたあとの可視の記録が残らない"
    assert "可視=はい" in later[0]


def test_a_bubble_whose_window_is_gone_is_reported_not_crashed(qapp, tmp_path, monkeypatch, caplog):
    """閉じた直後に1周が来ても落ちず、「消えていた」と残す。"""
    monkeypatch.setattr("familiar_agent.gui.Path.home", lambda: tmp_path)
    from familiar_agent.gui import ChatLog

    log = ChatLog(agent_label="パジュ")
    with caplog.at_level(logging.INFO, logger="familiar_agent.gui"):
        log.append_line("[パジュ] 目の前を見ますね。")
        log.deleteLater()
        del log
        qapp.processEvents()
        qapp.processEvents()
    later = [r.getMessage() for r in caplog.records if "描いたあと" in r.getMessage()]
    assert later and ("消えていた" in later[0] or "可視=" in later[0])
