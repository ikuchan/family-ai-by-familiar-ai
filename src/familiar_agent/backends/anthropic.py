"""AnthropicBackend（出-d-は）。Anthropic の公式 SDK を使う。

`backend.py` から**中身を変えずに**移した。約束は `core.llm_protocol.LLMBackend`。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any, cast

from .shared import (
    _supports_adaptive_thinking,
)
from .types import ToolCall, TurnResult

logger = logging.getLogger(__name__)


class AnthropicBackend:
    """Backend using the official Anthropic SDK."""

    def __init__(
        self,
        api_key: str,
        model: str,
        thinking_mode: str = "auto",
        thinking_budget: int = 10000,
        thinking_effort: str = "high",
    ) -> None:
        import anthropic

        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.thinking_mode = thinking_mode
        self.thinking_budget = thinking_budget
        self.thinking_effort = thinking_effort

    def _build_thinking_params(self, effort: str | None = None) -> dict:
        """Return thinking kwargs for the Anthropic API call.

        Per official docs (https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking):
        - adaptive thinking: no beta header needed (GA feature since Opus/Sonnet 4.6)
        - adaptive mode automatically enables interleaved thinking
        - extended mode on Sonnet 4.6 needs interleaved-thinking-2025-05-14 beta for interleaved support
        - extended mode on Opus 4.6 does NOT support interleaved thinking even with beta header

        Returns a dict that may contain:
          - "thinking": thinking config
          - "output_config": effort level (adaptive mode only)
          - "betas": list of beta header strings (extended mode on Sonnet 4.6 only)
        """
        mode = self.thinking_mode
        if mode == "auto":
            mode = "adaptive" if _supports_adaptive_thinking(self.model) else "disabled"

        # 呼び出しごとの effort（段階2 で軽量LLM が決める）。None ならインスタンスの設定。
        chosen_effort = effort or self.thinking_effort

        if mode == "adaptive":
            # No beta header required — adaptive thinking is GA on Opus 4.6 / Sonnet 4.6.
            # Interleaved thinking is automatically enabled in adaptive mode.
            params: dict = {"thinking": {"type": "adaptive"}}
            if chosen_effort != "high":  # "high" is the default; skip if default
                params["output_config"] = {"effort": chosen_effort}
            return params

        if mode == "extended":
            # budget_tokens is deprecated on Opus 4.6 / Sonnet 4.6 but still accepted.
            # interleaved-thinking-2025-05-14 beta enables interleaved thinking on Sonnet 4.6
            # only (Opus 4.6 extended mode does not support interleaved thinking).
            params = {"thinking": {"type": "enabled", "budget_tokens": self.thinking_budget}}
            if "sonnet-4" in self.model:
                params["betas"] = ["interleaved-thinking-2025-05-14"]
            return params

        # disabled (or unknown)
        return {}

    # ── message factories ─────────────────────────────────────────

    def make_user_message(self, content: str | list) -> dict:
        return {"role": "user", "content": content}

    def make_assistant_message(self, result: TurnResult, raw_content: Any) -> dict:  # noqa: ARG002
        return {"role": "assistant", "content": raw_content}

    def make_tool_results(
        self,
        tool_calls: list[ToolCall],
        results: list[tuple[str, str | None]],
    ) -> list[dict]:
        """Returns a one-element list containing the Anthropic tool_result user message."""
        content: list[dict[str, Any]] = []
        for tc, (text, image) in zip(tool_calls, results):
            result_content: list[dict[str, Any]] = [{"type": "text", "text": text or "(empty)"}]
            if image:
                result_content.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": image},
                    }
                )
            content.append({"type": "tool_result", "tool_use_id": tc.id, "content": result_content})
        if not content:
            content = [{"type": "text", "text": "(no tool results)"}]
        msgs: list[dict[str, Any]] = [{"role": "user", "content": content}]
        return msgs

    # ── API calls ─────────────────────────────────────────────────

    def _convert_tools(self, tool_defs: list[dict]) -> list[dict]:
        return tool_defs  # already in Anthropic format

    def _flatten_messages(self, messages: list) -> list[dict]:
        """Expand nested lists (from make_tool_results) into a flat message list."""
        flat: list[dict] = []
        for msg in messages:
            if isinstance(msg, list):
                flat.extend(msg)
            else:
                flat.append(msg)
        return flat

    @staticmethod
    def compact_images(messages: list[dict], keep_last: int = 3) -> list[dict]:
        """Strip base64 image data from old tool results, keeping the last `keep_last`.

        Human-like forgetting: the text description of what was seen is preserved;
        only the raw pixel data (base64) is dropped from older turns.

        Inspired by Claude Code's Dk() microcompact (KEEP_LAST=3).
        """
        import copy

        # Collect (msg_idx, tool_result_idx, sub_idx) for every image sub-item
        positions: list[tuple[int, int, int]] = []
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for j, item in enumerate(content):
                if not isinstance(item, dict) or item.get("type") != "tool_result":
                    continue
                for k, sub in enumerate(item.get("content", [])):
                    if isinstance(sub, dict) and sub.get("type") == "image":
                        positions.append((i, j, k))

        n_clear = max(0, len(positions) - keep_last)
        to_clear = positions[:n_clear]
        if not to_clear:
            return messages

        messages = copy.deepcopy(messages)
        for msg_i, item_j, sub_k in to_clear:
            messages[msg_i]["content"][item_j]["content"][sub_k] = {
                "type": "text",
                "text": "[image cleared]",
            }
        return messages

    @staticmethod
    def _build_system_param(system: str | tuple[str, str]) -> str | list[dict]:
        """Convert system prompt to Anthropic API format, adding cache_control when possible.

        If system is a (stable, variable) tuple, the stable block gets
        cache_control so it is reused across turns within the 5-minute window.
        If system is a plain string (e.g. from tests or other callers), pass as-is.
        """
        if not isinstance(system, tuple):
            return system
        stable, variable = system
        blocks: list[dict] = []
        if stable:
            blocks.append({"type": "text", "text": stable, "cache_control": {"type": "ephemeral"}})
        if variable:
            blocks.append({"type": "text", "text": variable})
        # Degenerate: if only one block, return as plain string (no cache_control needed)
        if len(blocks) == 1 and "cache_control" not in blocks[0]:
            return blocks[0]["text"]
        return blocks

    async def stream_turn(
        self,
        system: str | tuple[str, str],
        messages: list,
        tools: list[dict],
        max_tokens: int,
        on_text: Callable[[str], None] | None,
        effort: str | None = None,
    ) -> tuple[TurnResult, Any]:
        """Stream one agent turn. Returns (result, raw_content_for_assistant_message).

        `effort` は呼び出しごとの思考の深さ（段階2 で軽量LLM が決める）。None は既定。
        """
        from anthropic.types import MessageParam, ToolParam

        thinking_params = self._build_thinking_params(effort)
        betas = thinking_params.pop("betas", [])

        sys_param = self._build_system_param(system)
        # Build kwargs separately so we only add keys when they have meaningful values.
        # Passing thinking=None, output_config=None, or extra_headers=None are not valid.
        stream_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": sys_param,
            "tools": cast(list[ToolParam], self._convert_tools(tools)),
            "messages": cast(list[MessageParam], self._flatten_messages(messages)),
        }
        if betas:
            stream_kwargs["extra_headers"] = {"anthropic-beta": ",".join(betas)}
        if "thinking" in thinking_params:
            stream_kwargs["thinking"] = thinking_params["thinking"]
        if "output_config" in thinking_params:
            stream_kwargs["output_config"] = thinking_params["output_config"]
        flat_messages = self._flatten_messages(messages)
        flat_messages = self.compact_images(flat_messages)
        stream_kwargs["messages"] = cast(list[MessageParam], flat_messages)

        import anthropic as _anthropic

        _rate_limit_retries = 2
        for _attempt in range(_rate_limit_retries + 1):
            try:
                async with self.client.messages.stream(**stream_kwargs) as stream:  # type: ignore[arg-type]
                    async for chunk in stream.text_stream:
                        if on_text:
                            on_text(chunk)
                    response = await stream.get_final_message()
                break  # success
            except _anthropic.RateLimitError:
                if _attempt >= _rate_limit_retries:
                    raise
                _wait = 60 * (_attempt + 1)
                logger.warning(
                    "Anthropic 429 rate limit — waiting %ds before retry %d/%d",
                    _wait, _attempt + 1, _rate_limit_retries,
                )
                await asyncio.sleep(_wait)

        # ThinkingBlock has no .text attribute — hasattr check excludes it automatically
        text = "".join(b.text for b in response.content if hasattr(b, "text"))
        tool_calls = [
            ToolCall(id=b.id, name=b.name, input=b.input)
            for b in response.content
            if b.type == "tool_use"
        ]
        stop = "tool_use" if tool_calls else "end_turn"
        in_tok = getattr(response.usage, "input_tokens", 0) if response.usage else 0
        out_tok = getattr(response.usage, "output_tokens", 0) if response.usage else 0
        # Return response.content (including ThinkingBlocks) so interleaved thinking
        # tokens are round-tripped correctly in multi-turn conversations.
        return (
            TurnResult(
                stop_reason=stop,
                text=text,
                tool_calls=tool_calls,
                input_tokens=in_tok,
                output_tokens=out_tok,
            ),
            response.content,
        )

    async def warm_cache(self, stable_system: str) -> None:
        """Ping Anthropic with the stable system prompt to keep the 5-min cache TTL alive.

        Sends a minimal 1-token request.  The cache_read vs cache_creation token counts
        in the response tell us whether the block was already warm.
        """
        try:
            sys_param = self._build_system_param((stable_system, ""))
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=1,
                system=sys_param,  # type: ignore[arg-type]
                messages=[{"role": "user", "content": "."}],
            )
            usage = resp.usage
            if usage:
                read = getattr(usage, "cache_read_input_tokens", 0) or 0
                write = getattr(usage, "cache_creation_input_tokens", 0) or 0
                logger.debug("cache heartbeat ok — cache_read=%d cache_write=%d", read, write)
        except Exception as e:
            logger.debug("cache heartbeat skipped (non-critical): %s", e)

    async def complete(self, prompt: str, max_tokens: int) -> str:
        """Simple completion (no tools, no streaming) for utility calls."""
        try:
            logger.debug("complete() calling %s with %d chars", self.model, len(prompt))
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            from anthropic.types import TextBlock

            first = resp.content[0] if resp.content else None
            result = first.text.strip() if isinstance(first, TextBlock) else ""
            if not result:
                logger.warning(
                    "complete() empty response from %s: content=%s, stop=%s",
                    self.model,
                    resp.content,
                    resp.stop_reason,
                )
            return result
        except Exception as e:
            logger.warning("complete() failed: %s", e)
            return ""

    async def complete_with_image(self, prompt: str, image_b64: str, max_tokens: int = 512) -> str:
        """Vision completion — sends base64 JPEG alongside text prompt."""
        try:
            from anthropic.types import TextBlock

            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        }},
                    ],
                }],
            )
            first = resp.content[0] if resp.content else None
            return first.text.strip() if isinstance(first, TextBlock) else ""
        except Exception as e:
            logger.warning("complete_with_image() failed: %s", e)
            return ""
