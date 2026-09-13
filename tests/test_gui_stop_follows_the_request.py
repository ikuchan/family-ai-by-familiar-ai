"""GUI の停止ボタンは、求めが開いているあいだ有効で、押すと求めを打ち切る（環-j）。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.gui import FamiliarWindow


def _win():
    win = FamiliarWindow.__new__(FamiliarWindow)
    win._stop_btn = MagicMock()
    win._agent_running = False
    win._request_open = False
    win._agent_task = None
    win._cancel_requested = False
    win._log = MagicMock()
    win._input_queue = MagicMock(qsize=MagicMock(return_value=0))
    win._agent = MagicMock()
    win._agent.interrupt = AsyncMock()
    win._create_task = lambda coro: asyncio.get_event_loop().create_task(coro)
    return win


def test_the_button_stays_enabled_while_the_request_is_open() -> None:
    win = _win()
    FamiliarWindow._on_request_state(win, True)
    win._stop_btn.setEnabled.assert_called_with(True)
    FamiliarWindow._set_turn_ui_state(win, False)  # GUI のターンは終わった
    win._stop_btn.setEnabled.assert_called_with(True)  # 求めが開いているので有効のまま
    FamiliarWindow._on_request_state(win, False)
    win._stop_btn.setEnabled.assert_called_with(False)


def test_pressing_stop_interrupts_the_request_even_after_the_turn_ended() -> None:
    async def scenario():
        win = _win()
        FamiliarWindow._on_request_state(win, True)
        FamiliarWindow._cancel_turn(win, reason="user")
        await asyncio.sleep(0)
        return win

    win = asyncio.run(scenario())
    win._agent.interrupt.assert_awaited_once()
    assert any("[interrupted]" in str(c) for c in win._log.append_line.call_args_list)
