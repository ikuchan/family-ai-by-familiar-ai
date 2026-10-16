"""歩く道具 `walk` を撤去する（環-ak・2026-10-17）。

実機に脚は無い（`イベント駆動ループ` v0.135「確定した方針」：legs/walk 無し）。主LLM へ渡して呼ぶ口は
環-ab R-2（`816b594`）で撤去済みで、道具の本体・設定・表示・依存だけが残っていた。読むと
「動ける」に見える——次に触る者が誤る形である。道具と設定と依存も外した。

「walk」には別の意味の語（`ast.walk`、SQL の `WITH RECURSIVE walk` など）があるので、木全体を
正規表現では見ない。残っていてはいけない名前を 1 つずつ見る。
"""

from __future__ import annotations

import json
import pathlib

_ROOT = pathlib.Path(__file__).parent.parent
_SRC = _ROOT / "src/familiar_agent"
_GONE_KEYS = ("action_walk", "walk_timed", "walk_dir", "desire_prompt_explore")


def test_the_tool_file_is_gone() -> None:
    assert not (_SRC / "tools/mobility.py").exists()


def test_no_source_names_the_tool_or_its_config() -> None:
    hits = [
        f"{p.relative_to(_ROOT)}:{name}"
        for p in _SRC.rglob("*.py")
        for name in ("MobilityTool", "MobilityConfig", "TUYA_", "tinytuya")
        if name in p.read_text(encoding="utf-8")
    ]
    assert hits == []


def test_the_dependency_and_env_are_gone() -> None:
    assert "tinytuya" not in (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "TUYA_" not in (_ROOT / ".env.example").read_text(encoding="utf-8")


def test_no_table_knows_walk() -> None:
    from familiar_agent import agent
    from familiar_agent._ui_helpers import _I18N_ACTION_NAMES, ACTION_ICONS

    assert "walk" not in agent._TOOL_TIMEOUTS
    assert "walk" not in ACTION_ICONS
    assert "walk" not in _I18N_ACTION_NAMES


def test_no_locale_keeps_the_walk_strings() -> None:
    from familiar_agent._i18n import _T

    left = [
        f"{p.name}:{k}"
        for p in sorted((_SRC / "locales").glob("*.json"))
        for k in _GONE_KEYS
        if k in json.loads(p.read_text(encoding="utf-8"))
    ]
    assert left == []
    assert [k for k in _GONE_KEYS if k in _T] == []
