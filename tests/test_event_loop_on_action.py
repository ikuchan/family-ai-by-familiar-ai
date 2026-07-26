"""発話は `on_action("say", ...)` でも知らせる（GUI の表示経路）。

GUI は「発話は `on_action("say")` で来る」前提で作られている。素テキスト（`on_text`）は
say の**前**の途中経過を映すためのもので、say が出たら捨てられる。

```python
def on_text(chunk):
    if say_fired:
        return                     # say のあとの素テキストは捨てる
    self._stream.append_chunk(chunk)

def on_action(name, tool_input):
    if name == "say":
        say_fired = True
        self._stream.discard()     # それまでの素テキストも捨てる
        self._log.append_line(...)  # ここで画面に出る
```

イベント駆動ループは `on_text` にしか流しておらず、GUI では**何も表示されなかった**
（実機で観測）。ログ表示・ひとりごと判定・音声タグの除去は `on_action` の側に集まって
いるので、同じ約束で通知すればそれらがそのまま効く。GUI 側は変更しない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backend import ToolCall
from tests.test_event_loop import _agent, _turn

from familiar_agent.loop.event_loop import InformationProcessing


def _run_with_action(a, utterance="こんにちは"):
    actions: list[tuple[str, dict]] = []
    ip = InformationProcessing(a)
    ip.set_output(lambda _t: None, on_action=lambda n, i: actions.append((n, i)))
    asyncio.run(ip.run_iteration(utterance))
    return actions


def test_answer_is_reported_as_a_say_action():
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "やあ"})])])
    assert _run_with_action(a) == [("say", {"text": "やあ"})]


def test_filler_is_reported_as_a_say_action_too():
    # つなぎも発話なので、同じ経路で画面に出す。出さないと、調べているあいだ GUI が
    # 無反応に見える。
    a = _agent(stream_returns=[
        _turn([ToolCall(id="r", name="recall", input={"query": "q"})]),
        _turn([ToolCall(id="s", name="say", input={"text": "はい"})]),
    ])
    a._utility_backend.complete = AsyncMock(
        return_value='{"branch":"action","action":"recall","query":"q","text":"調べてみますね"}')
    actions = _run_with_action(a, "調べて")
    assert ("say", {"text": "調べてみますね"}) in actions


def test_output_can_be_registered_without_an_action_sink():
    # CUI は on_action を持たない。渡さなくても壊れないこと。
    ip = InformationProcessing(MagicMock())
    ip.set_output(lambda _t: None)
    assert ip._on_action is None
