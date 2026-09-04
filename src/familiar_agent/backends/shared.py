"""どのバックエンドも使う下請け（出-d-は）。

一時的な失敗の見分けと再試行、思考の深さを指定できるモデルの判定、道具の説明を
プロンプトへ畳む処理、返答から道具の呼び出しを読む処理、思考タグの濾し器。

**アンダースコアの名前のままにしてある。** テストが直接使っており（`_supports_adaptive_
thinking` は16箇所、`_retry_transient` と `_is_transient_error` は8箇所）、改名は
別の作業になる。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid

from .types import ToolCall

logger = logging.getLogger(__name__)


_TOOLS_PROMPT_HEADER = """\

---
[USING TOOLS]
You MUST use tools by outputting a <tool_call> block. This is the ONLY way to take actions.

RULE: When you want to use a tool, output EXACTLY this pattern and nothing after it:
<tool_call>{{"name": "...", "input": {{...}}}}</tool_call>

Then STOP. Do not write anything after the closing tag. The result will be given to you next.

CONCRETE EXAMPLES:
{examples}

Available tools:
{tools_desc}
[/USING TOOLS]
"""


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


_TRANSIENT_MARKERS = ("503", "unavailable", "429", "resource_exhausted", "high demand")


def _is_transient_error(exc: BaseException) -> bool:
    """一時的（再試行で回復しうる）エラーか。恒久エラー（400/401 等）は False。"""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (503, 429):
        return True
    s = str(exc).lower()
    return any(m in s for m in _TRANSIENT_MARKERS)


async def _retry_transient(fn, *, attempts: int, base_sec: float, label: str):
    """`fn`（async・無引数）を一時的エラーで指数バックオフ再試行する。恒久エラーは即送出。"""
    for i in range(attempts):
        try:
            return await fn()
        except Exception as e:  # noqa: BLE001
            if not _is_transient_error(e) or i == attempts - 1:
                raise
            delay = base_sec * (2 ** i)
            logger.warning(
                "%s transient error (retry %d/%d in %.1fs): %s", label, i + 1, attempts - 1, delay, e
            )
            if delay > 0:
                await asyncio.sleep(delay)


_ADAPTIVE_THINKING_MODELS = ("sonnet-4", "opus-4")


def _supports_adaptive_thinking(model: str) -> bool:
    """Return True if the model supports adaptive thinking (Sonnet 4.x / Opus 4.x)."""
    return any(m in model for m in _ADAPTIVE_THINKING_MODELS)


def _build_tools_system(system: str, tools: list[dict]) -> str:
    """Append tool descriptions + usage instructions to a system prompt."""
    if not tools:
        return system

    desc_lines = []
    example_lines = []
    for t in tools:
        props = t.get("input_schema", {}).get("properties", {})
        required = t.get("input_schema", {}).get("required", [])
        desc_lines.append(f"- {t['name']}: {t['description']}")

        example_input: dict = {}
        for k in required:
            prop = props.get(k, {})
            ptype = prop.get("type", "string")
            enum = prop.get("enum")
            if enum:
                example_input[k] = enum[0]
            elif ptype == "integer":
                example_input[k] = prop.get("default", 30)
            else:
                example_input[k] = f"<{k}>"
        example_json = json.dumps({"name": t["name"], "input": example_input}, ensure_ascii=False)
        example_lines.append(f"<tool_call>{example_json}</tool_call>")

    tools_desc = "\n".join(desc_lines)
    examples = "\n".join(example_lines)
    return system + _TOOLS_PROMPT_HEADER.format(tools_desc=tools_desc, examples=examples)


def _parse_tool_calls_from_text(text: str) -> list[ToolCall]:
    """Extract <tool_call> JSON blocks from model output."""
    tool_calls = []
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            data = json.loads(match.group(1).strip())
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    name=data["name"],
                    input=data.get("input", {}),
                )
            )
        except (json.JSONDecodeError, KeyError):
            logger.warning("Failed to parse tool_call: %s", match.group(1))
    return tool_calls


class _ThinkingTagFilter:
    """Strips <thinking>...</thinking> blocks from streamed Gemini text chunks.

    Maintains state across calls so tags split across chunk boundaries are
    handled correctly.  Also suppresses parts where part.thought is True
    (structured thinking API).
    """

    _OPEN = "<thinking>"
    _CLOSE = "</thinking>"

    def __init__(self) -> None:
        self._in_thinking = False
        self._buf = ""  # lookahead buffer for partial opening tags

    def feed(self, chunk: str) -> str:
        """Return the portion of *chunk* that should be emitted to the user."""
        self._buf += chunk
        out: list[str] = []
        while True:
            if not self._in_thinking:
                idx = self._buf.find(self._OPEN)
                if idx == -1:
                    # No opening tag in buffer — safe to emit all except a
                    # possible partial tag fragment at the tail.
                    tail = len(self._OPEN) - 1
                    if len(self._buf) > tail:
                        out.append(self._buf[:-tail])
                        self._buf = self._buf[-tail:]
                    break
                out.append(self._buf[:idx])
                self._buf = self._buf[idx + len(self._OPEN):]
                self._in_thinking = True
            else:
                idx = self._buf.find(self._CLOSE)
                if idx == -1:
                    break  # still inside block — buffer everything
                self._buf = self._buf[idx + len(self._CLOSE):]
                self._in_thinking = False
        return "".join(out)

    def flush(self) -> str:
        """Return any remaining buffered non-thinking text after streaming ends."""
        if self._in_thinking:
            return ""  # unclosed tag — discard
        result = self._buf
        self._buf = ""
        return result
