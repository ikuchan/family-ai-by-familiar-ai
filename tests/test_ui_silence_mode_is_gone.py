"""GUI・TUI の沈黙モードは撤去した（2026-09-16）。

`_silence_until` は GUI・TUI とも**書くだけでどこからも読まれていなかった**（初期化・
リセット・セットの 3 行だけ）。「黙って」の依頼はイベント駆動ループの `silence_request`
（`_delivery_block_reason`）が 1 箇所で見る。写しが残っていると、次に触る人が「GUI 側にも
沈黙の仕組みがある」と読んで二重に直す。
"""

from __future__ import annotations

from pathlib import Path

import familiar_agent._ui_helpers as helpers

_SRC = Path(__file__).resolve().parents[1] / "src" / "familiar_agent"


def test_the_helpers_no_longer_carry_a_silence_mode():
    assert not hasattr(helpers, "is_silence_request")
    assert not hasattr(helpers, "SILENCE_DURATION_SEC")


def test_neither_ui_keeps_a_silence_deadline():
    for name in ("gui.py", "tui.py"):
        src = (_SRC / name).read_text(encoding="utf-8")
        assert "_silence_until" not in src, name
        assert "is_silence_request" not in src, name
