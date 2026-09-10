"""求めの始め方を1つにした（環-e-に・段3）。

人の発話・情動・機器の3つは、どれも同じことをしていた。

    求めをリセットする → 来た事実を O へ書く → 求めの id を置く → 起点を控える
    → 手がかりを置く → 反復を回す

違うのは4点だけである。

| | 人の発話 | 情動 | 機器 |
|---|---|---|---|
| 起点の種別 | 発話 | 情動 | 機器 |
| 文面 | そのまま | `[内的な促し:軸] 促し` | `[種別] 内容` |
| `_utterance` | 発話 | 空 | 空 |
| 前にやること | 在席の印・出力先・打ち切り | — | — |
| 後にやること | — | — | 保留を配る（`release_pending`） |

**4つの状態（`iterations`・`request_text`・`trigger_kind`・`utterance`）の書き手が
3つから1つになる。** 器を作らずに、書き手そのものを減らす（`モジュール分割設計`）。
"""

from __future__ import annotations

import ast
from pathlib import Path

_LOOP = Path(__file__).parent.parent / "src/familiar_agent/loop/event_loop.py"


def _method(name: str) -> str:
    text = _LOOP.read_text(encoding="utf-8")
    lines = text.splitlines()
    cls = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.ClassDef) and n.name == "InformationProcessing"
    )
    m = next(
        x
        for x in cls.body
        if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name == name
    )
    return "\n".join(lines[m.lineno - 1 : m.end_lineno])


def test_there_is_one_place_that_begins_a_request():
    from familiar_agent.loop.event_loop import InformationProcessing

    assert hasattr(InformationProcessing, "_begin_request")


def test_the_three_entries_do_not_set_the_state_themselves():
    """**4つの状態を置くのは1箇所だけ**にする。"""
    for name in ("_utterance_iteration", "_begin_affect", "_begin_device"):
        body = _method(name)
        for field in (
            "_req.iterations",
            "_req.request_text",
            "_req.trigger_kind",
            "_req.utterance",
        ):
            assert f"self.{field} = " not in body, f"{name} が {field} を直に置いている"


def test_each_entry_goes_through_the_one_place():
    for name in ("_utterance_iteration", "_begin_affect", "_begin_device"):
        assert "self._begin_request(" in _method(name), name


def test_only_four_things_differ():
    """1箇所へ渡すのは、種別・文面・発話・手がかり だけ。"""
    import inspect

    from familiar_agent.loop.event_loop import InformationProcessing

    params = set(inspect.signature(InformationProcessing._begin_request).parameters)
    params.discard("self")
    assert params == {"kind", "text", "utterance"}, params
