"""写真を受ける担い手の口は、どれもシステム文を受ける。

以前はここで「調停が自分で見に行った帰りには、調停にも写真を渡す」（`イベント駆動ループ` v0.46）を確かめていた。
出-au 段 5-7a で、写真は読み取り（`scene.read_photo`）で状態として残し、調停には渡さない形へ改めた
（`test_the_photo_is_read_into_state.py`）。残るのは、写真を受ける口（読み取りと主LLM が使う）の形の確認だけである。
"""

from __future__ import annotations

import inspect


def test_every_vision_backend_accepts_a_system_prompt() -> None:
    from familiar_agent.backends.anthropic import AnthropicBackend
    from familiar_agent.backends.gemini import GeminiBackend
    from familiar_agent.backends.openai_compat import OpenAICompatibleBackend

    for cls in (AnthropicBackend, GeminiBackend, OpenAICompatibleBackend):
        params = inspect.signature(cls.complete_with_image).parameters
        assert "system" in params, cls.__name__
