"""主LLM・軽量LLM の実体（出-d-は）。

`backend.py` が 1,790 行に**6つのバックエンドを同居させていた**のを、モデルごとに
1ファイルへ分けた。**中身は変えていない**（行の範囲でそのまま移した）。

| ファイル | 中身 |
|---|---|
| `types.py` | `ToolCall`・`TurnResult`。どの実装からも独立している |
| `shared.py` | 一時的な失敗の再試行、思考の深さの判定、道具の説明の畳み込み、返答からの道具の読み取り、思考タグの濾し器 |
| `anthropic.py` ほか6つ | モデルごとの実装 |
| `factory.py` | `PLATFORM` と `MODEL` から実体を選ぶ |

約束は `core.llm_protocol.LLMBackend`（6つとも満たすことを型で確かめている）。
"""

from __future__ import annotations

from .anthropic import AnthropicBackend
from .cli import CLIBackend
from .factory import create_backend, create_scene_backend, create_utility_backend
from .gemini import GeminiBackend
from .glm import GLMBackend
from .kimi import KimiBackend
from .openai_compat import OpenAICompatibleBackend
from .shared import (
    _build_tools_system,
    _is_transient_error,
    _parse_tool_calls_from_text,
    _retry_transient,
    _supports_adaptive_thinking,
    _ThinkingTagFilter,
)
from .types import ToolCall, TurnResult

__all__ = [
    "AnthropicBackend", "CLIBackend", "GeminiBackend", "GLMBackend",
    "KimiBackend", "OpenAICompatibleBackend",
    "ToolCall", "TurnResult",
    "create_backend", "create_scene_backend", "create_utility_backend",
    "_build_tools_system", "_is_transient_error", "_parse_tool_calls_from_text",
    "_retry_transient", "_supports_adaptive_thinking", "_ThinkingTagFilter",
]
