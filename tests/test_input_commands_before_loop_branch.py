"""入力の解釈（/speaker・[名前]・/reload・思考モード）は、どちらのループでも効く。

これらは会話の中身ではなく入力の解釈なので、どのループを使うかと無関係。実装では
イベントループ分岐がこれらの処理より前で return しており、`/speaker パパ` がただの
発話として LLM へ流れていた（実機で「うん！」と返った）。分岐より前に置いて共通の
入口にする。個別に呼ぶ形にすると同じ処理が2箇所に散り、片方への足し忘れが繰り返される。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.agent import EmbodiedAgent as Agent


def _agent():
    a = MagicMock(spec=Agent)
    a.config = MagicMock()
    a._persons = MagicMock()
    a._persons.active_name = "あなた"
    a._persons.known_names = MagicMock(return_value=["パパ"])
    a._sync_pmm_speaker = AsyncMock()
    a._info_processing = MagicMock()
    a._info_processing.run_iteration = AsyncMock(return_value="LLM が答えた")
    a._ensure_event_loop = MagicMock()
    # 実物の解釈を通す（spec の MagicMock は None でない値を返して早期 return する）。
    a._handle_speaker_command = lambda ui: Agent._handle_speaker_command(a, ui)
    a._handle_reload_command = MagicMock(return_value=None)
    a._handle_thinking_command = MagicMock(return_value=None)
    return a


def test_speaker_command_is_handled_on_the_event_loop_path():
    a = _agent()
    reply = asyncio.run(Agent.run(a, "/speaker パパ"))
    a._persons.set_active.assert_called_once_with("パパ")
    assert "パパ" in reply
    # コマンドなので LLM を起こさない。
    a._info_processing.run_iteration.assert_not_awaited()


def test_speaker_prefix_is_stripped_on_the_event_loop_path():
    a = _agent()
    asyncio.run(Agent.run(a, "[たいき] こんにちは"))
    a._persons.set_active.assert_called_once_with("たいき")
    # 前置きを外した本文だけが反復へ渡る。
    assert a._info_processing.run_iteration.await_args.args[0] == "こんにちは"
