"""環-bb 三段目：状態図の窓（`gui_state_chart`）。図を描く部品（QtWebEngine）は起こさず、読む口だけを確かめる。"""

from __future__ import annotations

import logging


def test_recent_transitions_come_from_the_log_tail_newest_first(tmp_path):
    from familiar_agent.gui_state_chart import recent_transitions

    log = tmp_path / "app.log"
    rows = [
        f"2026-11-21 20:00:{i:02d},000 [v0.x] [INFO] familiar_agent.loop.paju_state: 状態図 ev{i} で移った（頭=何もしていない）"
        for i in range(30)
    ]
    log.write_text("\n".join(["ほかの行", *rows]) + "\n", encoding="utf-8")
    got = recent_transitions(log, limit=3)
    assert [e for _at, e, _d in got] == ["ev29", "ev28", "ev27"]
    assert got[0][0] == "2026-11-21 20:00:29"


def test_a_missing_log_gives_nothing(tmp_path):
    from familiar_agent.gui_state_chart import recent_transitions

    assert recent_transitions(tmp_path / "無い.log") == []
    assert recent_transitions(None) == []


def test_the_log_path_is_read_from_the_root_file_handler(tmp_path):
    from familiar_agent.gui_state_chart import log_path

    h = logging.FileHandler(tmp_path / "app.log")
    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers = [logging.StreamHandler(), h]  # 本番は main.py が付けたファイル出力 1 つ
    try:
        assert log_path() == tmp_path / "app.log"
    finally:
        root.handlers = saved
        h.close()


def test_mermaid_is_bundled_with_its_license():
    from familiar_agent.gui_state_chart import _ASSETS

    assert (_ASSETS / "mermaid.min.js").stat().st_size > 1_000_000
    assert "MIT License" in (_ASSETS / "LICENSE").read_text(encoding="utf-8")
