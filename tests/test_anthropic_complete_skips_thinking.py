"""`AnthropicBackend.complete()` は思考ブロックを飛ばして本文をつなぐ（環-l・2026-09-15）。

Sonnet 5 は返りの先頭に `ThinkingBlock` を置くことがある。`content[0]` だけを見ると空文字になり、
REST の層 1 ①が 69 回すべて「返りを読めなかった」（806 秒・書いた 0）、層 4 も「capabilities が空」。
本番の 2026-06-13 の 60 件で再現した。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from anthropic.types import TextBlock, ThinkingBlock

from familiar_agent.backends.anthropic import AnthropicBackend


def _backend(content: list, stop: str = "end_turn") -> AnthropicBackend:
    b = AnthropicBackend.__new__(AnthropicBackend)
    b.model = "claude-sonnet-5"
    b.client = SimpleNamespace(
        messages=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(content=content, stop_reason=stop))
        )
    )
    return b


def _think() -> ThinkingBlock:
    return ThinkingBlock(type="thinking", thinking="", signature="x")


def _text(t: str) -> TextBlock:
    return TextBlock(type="text", text=t)


def test_the_text_after_a_thinking_block_is_returned():
    b = _backend([_think(), _text('{"episode": "…"}')])
    assert asyncio.run(b.complete("p", max_tokens=10)) == '{"episode": "…"}'


def test_several_text_blocks_are_joined_in_order():
    b = _backend([_text("前半"), _think(), _text("後半")])
    assert asyncio.run(b.complete("p", max_tokens=10)) == "前半後半"


def test_only_thinking_is_empty_and_warned(caplog):
    b = _backend([_think()])
    with caplog.at_level("WARNING"):
        assert asyncio.run(b.complete("p", max_tokens=10)) == ""
    assert any("empty response" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_the_image_completion_skips_thinking_too():
    b = _backend([_think(), _text("見えたもの：椅子")])
    assert await b.complete_with_image("p", "aGVsbG8=", max_tokens=10) == "見えたもの：椅子"


def test_complete_disables_thinking_so_the_budget_goes_to_the_text():
    """利用の呼び出し（REST の各層・発話前の検査）は思考を切る（2026-09-16 04:31 実機）。

    思考の指定を渡さないと Sonnet 5 は適応的思考を働かせ、層 2（2,000）・層 4（2,500）の
    予算を思考で使い切って `stop=max_tokens`・本文ゼロで返った。
    """
    b = _backend([_text("ok")])
    asyncio.run(b.complete("p", max_tokens=10))
    kwargs = b.client.messages.create.call_args.kwargs
    assert kwargs.get("thinking") == {"type": "disabled"}
    b = _backend([_text("ok")])
    asyncio.run(b.complete_with_image("p", "aGVsbG8=", max_tokens=10))
    assert b.client.messages.create.call_args.kwargs.get("thinking") == {"type": "disabled"}


def _backend_for(model: str) -> AnthropicBackend:
    b = _backend([_text("ok")])
    b.model = model
    return b


def test_sonnet_5_5_turns_thinking_off_with_between_tools():
    """Sonnet 5.5 は `disabled` を 400 で断り、代わりに `between_tools` を受け取る（環-al・2026-10-17）。

    公式の文書（Troubleshooting thinking）と実測で、`between_tools` は返事の前に考えない。予算 120 でも
    返事が空にならなかった（adaptive low は 5 回中 2 回、考えで予算を使い切って空だった）。
    """
    b = _backend_for("claude-sonnet-5-5")
    asyncio.run(b.complete("p", max_tokens=10))
    assert b.client.messages.create.call_args.kwargs.get("thinking") == {"type": "between_tools"}
    b = _backend_for("claude-sonnet-5-5")
    asyncio.run(b.complete_with_image("p", "aGVsbG8=", max_tokens=10))
    assert b.client.messages.create.call_args.kwargs.get("thinking") == {"type": "between_tools"}


def test_other_models_keep_disabled():
    """**反証の側**：`between_tools` を受け取るのは Sonnet 5.5 だけ。ほかに送れば 400 になる。"""
    for model in ("claude-sonnet-5", "claude-opus-5", "claude-haiku-5-5"):
        b = _backend_for(model)
        asyncio.run(b.complete("p", max_tokens=10))
        assert b.client.messages.create.call_args.kwargs.get("thinking") == {"type": "disabled"}, (
            model
        )
