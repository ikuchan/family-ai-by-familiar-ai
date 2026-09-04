"""主LLM・軽量LLM の実体（**移行中の薄い層**）。

**中身は `backends/` へ移した**（出-d-は-1）。ここは既存の `from .backend import ...` を
壊さないための再輸出だけである。**新しい import は `backends/` から書くこと。**

この層は 出-d-は-3 で消す。消したときに import エラーが出れば、それが「旧い経路がまだ
残っている」ことの証明になる（`旧名の grep が0件` と同じ検算）。
"""

from __future__ import annotations

from .backends import (  # noqa: F401
    AnthropicBackend,
    CLIBackend,
    GeminiBackend,
    GLMBackend,
    KimiBackend,
    OpenAICompatibleBackend,
    ToolCall,
    TurnResult,
    _build_tools_system,
    _is_transient_error,
    _parse_tool_calls_from_text,
    _retry_transient,
    _supports_adaptive_thinking,
    _ThinkingTagFilter,
    create_backend,
    create_scene_backend,
    create_utility_backend,
)
