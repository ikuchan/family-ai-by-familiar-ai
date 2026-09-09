"""話すことと、反復を閉じることを分けた（環-e-に・段4）。

`_speak` は「話す」動作なのに、**反復を閉じるところまで持っていた**。docstring 自身が
「発話して反復を閉じる」と2つを述べており、結末の語（沈黙・保留・発話）を決めるのも
`_speak` だった。`memories` を受け取るのも、自分が使うためではなく `_finish` へ渡すため
だけだった。

核は殻を呼び返さない（Bernhardt, *Functional Core / Imperative Shell*・
`モジュール分割設計`）。**閉じるのは、求めの寿命を持っている側（`_iterate`）の仕事**である。

`_speak` は「何が起きたか」＝ (実際に出した文, 結末) を返すだけにする。
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

_LOOP = Path(__file__).parent.parent / "src/familiar_agent/loop/event_loop.py"


def _method_src(name: str) -> str:
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


def _ip(blocked: str = ""):
    from familiar_agent.loop.event_loop import InformationProcessing

    ip = InformationProcessing.__new__(InformationProcessing)
    ip._delivery_block_reason = MagicMock(return_value=blocked)
    ip._dif = MagicMock(speak=AsyncMock())
    ip._emit = MagicMock()
    ip._hold_speech = AsyncMock()
    ip._finish = AsyncMock()
    return ip


# ── 返す ───────────────────────────────────────────────────────────────────


def test_speaking_returns_what_happened():
    ip = _ip()
    assert asyncio.run(ip._speak("はい")) == ("はい", "発話")
    ip._dif.speak.assert_awaited_once_with("はい")
    ip._emit.assert_called_once_with("はい")


def test_nothing_to_say_is_silence():
    ip = _ip()
    assert asyncio.run(ip._speak("")) == ("", "沈黙")
    ip._dif.speak.assert_not_awaited()


def test_a_blocked_delivery_is_held():
    ip = _ip(blocked="聞く相手が居ない")
    assert asyncio.run(ip._speak("はい")) == ("", "保留")
    ip._hold_speech.assert_awaited_once_with("はい", "聞く相手が居ない")
    ip._dif.speak.assert_not_awaited()


# ── 閉じない ───────────────────────────────────────────────────────────────


def test_speaking_never_closes_the_turn():
    ip = _ip()
    for text, blocked in (("はい", ""), ("", ""), ("はい", "聞く相手が居ない")):
        ip = _ip(blocked=blocked)
        asyncio.run(ip._speak(text))
        ip._finish.assert_not_awaited()


def test_the_source_shows_it_too():
    """docstring は経緯として `_finish` を語るので、**コードの行だけ**を見る。"""
    import io
    import tokenize

    body = _method_src("_speak")
    code = " ".join(
        t.string
        for t in tokenize.generate_tokens(io.StringIO(body.lstrip()).readline)
        if t.type not in (tokenize.COMMENT, tokenize.STRING)
    )
    assert "_finish" not in code


def test_speaking_no_longer_takes_the_memories():
    """`memories` は `_finish` へ渡すためだけに受け取っていた。"""
    from familiar_agent.loop.event_loop import InformationProcessing

    params = set(inspect.signature(InformationProcessing._speak).parameters)
    params.discard("self")
    assert params == {"text"}, params


def test_only_the_iteration_closes_the_turn():
    """`_finish` を呼ぶのは `_iterate` だけ。"""
    text = _LOOP.read_text(encoding="utf-8")
    lines = text.splitlines()
    cls = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.ClassDef) and n.name == "InformationProcessing"
    )
    callers = set()
    for m in cls.body:
        if not isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) or m.name == "_finish":
            continue
        if "self._finish(" in "\n".join(lines[m.lineno - 1 : m.end_lineno]):
            callers.add(m.name)
    # 環-h・段ろ で、主LLM の返りを実行する部分を `_act_on_decision` へ出した。
    # **閉じるのは、その決定を実行している側**である（話す動作ではない）。
    assert callers == {"_iterate", "_act_on_decision"}, callers
