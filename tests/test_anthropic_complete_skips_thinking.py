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
    for model in ("claude-sonnet-5", "claude-haiku-5-5"):  # Opus 5 系は effort medium（下の試験）
        b = _backend_for(model)
        asyncio.run(b.complete("p", max_tokens=10))
        assert b.client.messages.create.call_args.kwargs.get("thinking") == {"type": "disabled"}, (
            model
        )


def test_opus_complete_sends_medium_effort_and_more_tokens_instead_of_disabled():
    """Opus 5 系は思考を切る指定（disabled）を 400 で断る。思考の指定を送らず effort medium にし、考える分として
    max_tokens に 2000 を足す（2026-11-14 本人承認・Opus 5.5 の実測で考える分は最大およそ 450）。"""
    b = _backend_for("claude-opus-5-5")
    asyncio.run(b.complete("p", max_tokens=120))
    kwargs = b.client.messages.create.call_args.kwargs
    assert "thinking" not in kwargs
    assert kwargs["output_config"] == {"effort": "medium"}
    assert kwargs["max_tokens"] == 2120


def test_opus_image_completion_does_the_same():
    b = _backend_for("claude-opus-5-5")
    asyncio.run(b.complete_with_image("p", "aGVsbG8=", max_tokens=100))
    kwargs = b.client.messages.create.call_args.kwargs
    assert "thinking" not in kwargs
    assert (kwargs["output_config"], kwargs["max_tokens"]) == ({"effort": "medium"}, 2100)


def test_hitting_max_tokens_is_logged(caplog):
    """上限で止まったことを INFO に 1 行残す（足した 2000 で足りているかを家で数えるため）。"""
    b = _backend([_text("途中まで")], stop="max_tokens")
    b.model = "claude-opus-5-5"
    with caplog.at_level("INFO"):
        asyncio.run(b.complete("p", max_tokens=120))
    assert any("上限で止まった" in r.message for r in caplog.records)


def test_hitting_max_tokens_in_an_image_completion_is_logged_too(caplog):
    """写真の呼び出しも上限で止まったら INFO に 1 行残す（足した 2000 を家で数えるとき写真も入れる・2026-11-15 本人「ア」）。"""
    b = _backend([_text("途中まで")], stop="max_tokens")
    b.model = "claude-opus-5-5"
    with caplog.at_level("INFO"):
        asyncio.run(b.complete_with_image("p", "aGVsbG8=", max_tokens=100))
    assert any("上限で止まった" in r.message for r in caplog.records)


def test_the_system_text_of_a_utility_call_is_marked_for_caching():
    """軽量LLM の呼び出しでも、システムの文にキャッシュの印を付ける（出-dx・2026-11-24）。

    11/21〜23 の haiku の呼び出し 301 回（入力 205 万トークン）で、キャッシュの読み書きが 0 だった。システムの文を文字列のまま
    送っていたため。調停の書き手のシステムの文は主LLM と同じ変わらない部分で、呼び出しの 80% が前から 5 分以内に続く。
    """
    b = _backend([_text("ok")])
    assert asyncio.run(b.complete("問い", 10, system="立ち位置")) == "ok"
    sent = b.client.messages.create.await_args.kwargs["system"]
    assert sent == [{"type": "text", "text": "立ち位置", "cache_control": {"type": "ephemeral"}}]


def test_no_system_text_means_no_system_block():
    """**反証の側**：システムの文が無い呼び出しには、何も足さない。"""
    b = _backend([_text("ok")])
    asyncio.run(b.complete("問い", 10))
    assert "system" not in b.client.messages.create.await_args.kwargs


def test_the_system_text_of_a_photo_call_is_marked_for_caching():
    """写真を見る軽量LLM の呼び出し（`complete_with_image`）も、システムの文にキャッシュの印を付ける（出-dx）。"""
    b = _backend([_text("ok")])
    asyncio.run(b.complete_with_image("何が見える？", "aGVsbG8=", 10, system="立ち位置"))
    sent = b.client.messages.create.await_args.kwargs["system"]
    assert sent == [{"type": "text", "text": "立ち位置", "cache_control": {"type": "ephemeral"}}]
