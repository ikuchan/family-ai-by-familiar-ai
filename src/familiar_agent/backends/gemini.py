"""GeminiBackend（出-d-は）。Gemini を使う。

`backend.py` から**中身を変えずに**移した。約束は `core.llm_protocol.LLMBackend`。
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable
from typing import Any

from .shared import (
    _retry_transient,
    _ThinkingTagFilter,
)
from .types import ToolCall, TurnResult

logger = logging.getLogger(__name__)


class GeminiBackend:
    """Backend using the official Google Generative AI SDK (google-generativeai).

    Advantages over OpenAI-compatible endpoint:
    - Native function calling without format hacks
    - thinkingBudget can be set properly (no thinking token leakage)
    - Access to Gemini-specific features
    """

    def __init__(self, api_key: str, model: str) -> None:
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self.model = model
        # 一時的エラー（Google 側 503 過負荷等）の指数バックオフ・リトライ設定。
        try:
            self._retry_attempts = int(os.environ.get("GEMINI_RETRY_ATTEMPTS", "3"))
        except ValueError:
            self._retry_attempts = 3
        try:
            self._retry_base = float(os.environ.get("GEMINI_RETRY_BASE_SEC", "0.5"))
        except ValueError:
            self._retry_base = 0.5

    # ── message factories ─────────────────────────────────────────

    def make_user_message(self, content: str | list) -> dict:
        if isinstance(content, str):
            return {"role": "user", "parts": [{"text": content}]}
        parts: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                parts.append({"text": item})
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append({"text": item["text"]})
                elif item.get("type") == "image":
                    src = item["source"]
                    parts.append(
                        {"inline_data": {"mime_type": src["media_type"], "data": src["data"]}}
                    )
        return {"role": "user", "parts": parts}

    def make_assistant_message(self, result: TurnResult, raw_content: Any) -> dict:  # noqa: ARG002
        return raw_content  # already Gemini-format Content dict

    def make_tool_results(
        self,
        tool_calls: list[ToolCall],
        results: list[tuple[str, str | None]],
    ) -> list[dict]:
        parts: list[dict[str, Any]] = []
        for tc, (text, image) in zip(tool_calls, results):
            parts.append({"function_response": {"name": tc.name, "response": {"result": text}}})
            if image:
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image}})
        return [{"role": "user", "parts": parts}]

    # ── API calls ─────────────────────────────────────────────────

    # JSON Schema keywords unsupported by Gemini FunctionDeclaration
    _GEMINI_UNSUPPORTED_SCHEMA_KEYS: frozenset[str] = frozenset({
        "exclusiveMaximum", "exclusiveMinimum",
        "$schema", "$id", "$ref", "$comment",
        "additionalItems", "contains", "patternProperties",
        "dependencies", "propertyNames", "const",
        "if", "then", "else", "allOf", "anyOf", "oneOf", "not",
        "examples", "readOnly", "writeOnly",
    })

    @classmethod
    def _sanitize_schema(cls, schema: dict) -> dict:
        """Recursively remove JSON Schema keywords unsupported by Gemini."""
        # 値は dict にも list にも素の値にもなる（JSON Schema をそのまま写すため）。
        result: dict[Any, Any] = {}
        for k, v in schema.items():
            if k in cls._GEMINI_UNSUPPORTED_SCHEMA_KEYS:
                continue
            if isinstance(v, dict):
                result[k] = cls._sanitize_schema(v)
            elif isinstance(v, list):
                result[k] = [
                    cls._sanitize_schema(item) if isinstance(item, dict) else item
                    for item in v
                ]
            else:
                result[k] = v
        return result

    def _convert_tools(self, tool_defs: list[dict]) -> list:
        types = self._types
        declarations = []
        for t in tool_defs:
            try:
                declarations.append(types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=self._sanitize_schema(t["input_schema"]),
                ))
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "Skipping tool %s for Gemini (schema error): %s", t["name"], e
                )
        return [types.Tool(function_declarations=declarations)]

    @staticmethod
    def _to_gemini_message(msg: dict) -> dict:
        """Convert a non-Gemini format message to Gemini parts format.

        Silently fixes messages in Anthropic/OpenAI format (role+content)
        that occasionally leak in when the utility backend is not Gemini.
        Logs a warning so the root cause can be traced.
        """
        if "parts" in msg or "content" not in msg:
            return msg
        role = msg.get("role", "user")
        content = msg["content"]
        if isinstance(content, str):
            parts: list = [{"text": content}]
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append({"text": block.get("text", "")})
        else:
            parts = [{"text": str(content)}]
        gemini_role = "model" if role == "assistant" else role
        logger.warning(
            "Gemini: non-Gemini message coerced (role=%s). "
            "Check utility-backend message isolation.",
            role,
        )
        return {"role": gemini_role, "parts": parts}

    @staticmethod
    def _to_gemini_message_silent(msg: dict) -> dict:
        """Convert an Anthropic-format message to Gemini format without logging.

        Use this for intentional bulk conversion (e.g. when swapping to the
        utility backend for internal desire turns). _to_gemini_message() is
        kept for accidental-leakage detection and still logs a warning.
        """
        if "parts" in msg or "content" not in msg:
            return msg
        role = msg.get("role", "user")
        content = msg["content"]
        if isinstance(content, str):
            parts: list = [{"text": content}]
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append({"text": block.get("text", "")})
        else:
            parts = [{"text": str(content)}]
        gemini_role = "model" if role == "assistant" else role
        return {"role": gemini_role, "parts": parts}

    @staticmethod
    def convert_messages_to_gemini_format(messages: list[dict]) -> list[dict]:
        """Return a new list with all messages converted to Gemini format.

        Intended for intentional conversion when switching to the Gemini
        utility backend. Does not log warnings.
        """
        return [GeminiBackend._to_gemini_message_silent(m) for m in messages]

    def _flatten_messages(self, messages: list) -> list[dict]:
        flat: list[dict] = []
        for msg in messages:
            if isinstance(msg, list):
                flat.extend(self._to_gemini_message(m) for m in msg if isinstance(m, dict))
            elif isinstance(msg, dict):
                flat.append(self._to_gemini_message(msg))
        return flat

    async def stream_turn(
        self,
        system: str | tuple[str, str],
        messages: list,
        tools: list[dict],
        max_tokens: int,
        on_text: Callable[[str], None] | None,
        effort: str | None = None,   # 署名を揃えるだけ（未対応）
    ) -> tuple[TurnResult, Any]:
        if isinstance(system, tuple):
            system = "\n\n---\n\n".join(s for s in system if s)
        types = self._types
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=self._convert_tools(tools) if tools else None,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        contents = self._flatten_messages(messages)

        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []
        raw_parts: list = []
        _tf = _ThinkingTagFilter()

        async for chunk in await self._client.aio.models.generate_content_stream(
            model=self.model,
            contents=contents,  # type: ignore[arg-type]
            config=config,
        ):
            if not chunk.candidates:
                continue
            content = chunk.candidates[0].content
            if content is None or content.parts is None:
                continue
            for part in content.parts:
                raw_parts.append(part)
                if part.text and not getattr(part, "thought", False):
                    filtered = _tf.feed(part.text)
                    if filtered:
                        text_chunks.append(filtered)
                        if on_text:
                            on_text(filtered)
                if part.function_call:
                    fc = part.function_call
                    if fc.name is None:
                        continue
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name=fc.name,
                            input=dict(fc.args or {}),
                        )
                    )

        tail = _tf.flush()
        if tail:
            text_chunks.append(tail)
            if on_text:
                on_text(tail)
        text = "".join(text_chunks)
        stop = "tool_use" if tool_calls else "end_turn"
        raw_assistant = {"role": "model", "parts": raw_parts}
        return TurnResult(stop_reason=stop, text=text, tool_calls=tool_calls), raw_assistant

    async def complete(self, prompt: str, max_tokens: int) -> str:
        types = self._types

        async def _call() -> str:
            resp = await self._client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return (resp.text or "").strip()

        try:
            return await _retry_transient(
                _call, attempts=self._retry_attempts, base_sec=self._retry_base,
                label="gemini.complete",
            )
        except Exception as e:
            logger.warning("complete() failed: %s", e)
            return ""

    async def complete_with_image(self, prompt: str, image_b64: str, max_tokens: int = 512) -> str:
        """Vision completion — sends base64 JPEG alongside text prompt."""
        types = self._types

        async def _call() -> str:
            resp = await self._client.aio.models.generate_content(
                model=self.model,
                # SDK の型定義は `inline_data` を含む素の dict を受け付けない形になって
                # いるが、実行時には受け付ける（画像を渡す経路は実機で動いている）。
                contents=[{  # type: ignore[arg-type]
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                    ],
                }],
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return (resp.text or "").strip()

        try:
            return await _retry_transient(
                _call, attempts=self._retry_attempts, base_sec=self._retry_base,
                label="gemini.complete_with_image",
            )
        except Exception as e:
            logger.warning("complete_with_image() failed: %s", e)
            return ""
