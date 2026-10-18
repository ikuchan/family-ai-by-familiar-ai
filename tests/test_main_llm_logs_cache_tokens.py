"""主LLM の呼び出しごとに、キャッシュの読み・書きと入力・出力のトークン数を 1 行残す（出-bs の計測・2026-10-19）。

Anthropic の管理画面に「プロンプトキャッシュの当たりが低い」と出たが、呼び出しごとの数を残していないので、
どこで外れているか（5 分を超えた間隔か・道具の並びの違いか・印の無い REST の呼び出しか）を数えられなかった。
数だけを出し、渡した文の中身は出さない。
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from anthropic.types import TextBlock

from familiar_agent.backends.anthropic import AnthropicBackend

_LOGGER = "familiar_agent.backends.anthropic"
_SECRET = "目印の文はログに出ない"


def _usage(read=0, write=0, inp=0, out=0):
    return SimpleNamespace(
        cache_read_input_tokens=read,
        cache_creation_input_tokens=write,
        input_tokens=inp,
        output_tokens=out,
    )


def _backend() -> AnthropicBackend:
    b = AnthropicBackend.__new__(AnthropicBackend)
    b.model = "claude-haiku-4-5-20251001"  # 思考を付けない型
    b.thinking_mode = "disabled"
    b.thinking_budget = 10000
    b.thinking_effort = "high"
    b.client = MagicMock()
    return b


def _with_stream(b: AnthropicBackend, usage) -> None:
    final = SimpleNamespace(
        stop_reason="end_turn", content=[SimpleNamespace(type="text", text="はい")], usage=usage
    )
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=cm)
    cm.__aexit__ = AsyncMock(return_value=False)

    async def _text():
        yield "はい"

    cm.text_stream = _text()
    cm.get_final_message = AsyncMock(return_value=final)
    b.client.messages.stream = MagicMock(return_value=cm)


def _with_create(b: AnthropicBackend, usage) -> None:
    resp = SimpleNamespace(
        content=[TextBlock(type="text", text="はい")], stop_reason="end_turn", usage=usage
    )
    b.client.messages.create = AsyncMock(return_value=resp)


_TOOLS = [
    {"name": "say", "description": "話す", "input_schema": {"type": "object", "properties": {}}},
    {"name": "recall", "description": "思い出す", "input_schema": {"type": "object"}},
]


def _lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "主LLM トークン" in r.getMessage()]


def test_the_logger_is_heard(caplog):
    """捕まえ損ねで落ちているのではないことを先に見る（RED の反証）。"""
    caplog.set_level(logging.INFO, logger=_LOGGER)
    logging.getLogger(_LOGGER).info("聞こえる")
    assert "聞こえる" in caplog.text


def test_a_turn_logs_its_cache_tokens_and_tools(caplog):
    caplog.set_level(logging.INFO, logger=_LOGGER)
    b = _backend()
    _with_stream(b, _usage(read=3000, write=0, inp=500, out=40))
    asyncio.run(
        b.stream_turn(
            system=(_SECRET, "可変部"),
            messages=[{"role": "user", "content": _SECRET}],
            tools=_TOOLS,
            max_tokens=256,
            on_text=None,
        )
    )
    (line,) = _lines(caplog)
    for part in ("呼び=反復", "読み=3000", "書き=0", "入力=500", "出力=40", "道具=2"):
        assert part in line, line
    assert _SECRET not in caplog.text


def test_a_utility_call_logs_too(caplog):
    caplog.set_level(logging.INFO, logger=_LOGGER)
    b = _backend()
    _with_create(b, _usage(read=0, write=1200, inp=80, out=300))
    assert asyncio.run(b.complete(_SECRET, 512)) == "はい"
    (line,) = _lines(caplog)
    for part in ("呼び=利用", "読み=0", "書き=1200", "入力=80", "出力=300"):
        assert part in line, line
    assert _SECRET not in caplog.text


def test_an_image_call_logs_too(caplog):
    caplog.set_level(logging.INFO, logger=_LOGGER)
    b = _backend()
    _with_create(b, _usage(read=0, write=0, inp=1500, out=60))
    asyncio.run(b.complete_with_image(_SECRET, "QUJD"))
    (line,) = _lines(caplog)
    assert "呼び=写真" in line and "入力=1500" in line, line


def test_a_reply_without_usage_logs_zeros_and_does_not_fail(caplog):
    """**反証の側**：`usage` の無い返り（古い偽物・値が None）でも落ちず、0 で出す。"""
    caplog.set_level(logging.INFO, logger=_LOGGER)
    b = _backend()
    _with_create(b, None)
    assert asyncio.run(b.complete("問い", 512)) == "はい"
    _with_create(b, _usage(read=None, write=None, inp=None, out=None))
    assert asyncio.run(b.complete("問い", 512)) == "はい"
    lines = _lines(caplog)
    assert len(lines) == 2
    for line in lines:
        assert "読み=0" in line and "書き=0" in line and "入力=0" in line, line
