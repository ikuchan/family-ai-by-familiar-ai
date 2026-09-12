"""調停が自分で見に行った帰りには、調停にも写真を渡す（`イベント駆動ループ` v0.46）。

即席のラベル（YOLO）はこの部屋で `bench`／`chair、dining table`／0 件と揺れ、調停は材料不足で
毎回 full へ倒れた（実機 2026-09-12・4 回とも）。写真そのものがあれば 1 回の調停で light に
答えられる。写真を受けられない調停（`complete_with_image` の無い担い手）は文字だけで進む。
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.arbiter import arbitrate


def _backend(*, vision: bool):
    b = MagicMock(spec=["complete", "complete_with_image"] if vision else ["complete"])
    b.complete = AsyncMock(return_value='{"branch": "full"}')
    if vision:
        b.complete_with_image = AsyncMock(
            return_value='{"branch": "light", "text": "机が見えます"}'
        )
    return b


def test_with_a_photo_the_arbiter_is_asked_through_the_vision_call() -> None:
    b = _backend(vision=True)
    d = asyncio.run(
        arbitrate(b, utterance="何が見える？", workspace_ctx="", can_see=True, image_b64="QUJD")
    )
    assert d.branch == "light" and d.text == "机が見えます"
    assert b.complete_with_image.called and not b.complete.called
    kwargs = b.complete_with_image.call_args.kwargs
    assert kwargs.get("system"), "人格・家族はシステム文で渡す（文字だけのときと同じ）"
    assert b.complete_with_image.call_args.args[1] == "QUJD"


def test_the_prompt_says_a_photo_is_attached() -> None:
    b = _backend(vision=True)
    asyncio.run(arbitrate(b, utterance="x", workspace_ctx="", can_see=True, image_b64="QUJD"))
    prompt = b.complete_with_image.call_args.args[0]
    assert "写真を添えた" in prompt
    assert "写真そのものは主LLM にだけ" not in prompt, "写真があるのに『主LLM だけ』と言わない"


def test_without_a_photo_nothing_changes() -> None:
    b = _backend(vision=True)
    asyncio.run(arbitrate(b, utterance="x", workspace_ctx="", can_see=True))
    assert b.complete.called and not b.complete_with_image.called


def test_a_backend_without_eyes_gets_the_text_only() -> None:
    b = _backend(vision=False)
    d = asyncio.run(arbitrate(b, utterance="x", workspace_ctx="", can_see=True, image_b64="QUJD"))
    assert b.complete.called and d.branch == "full"


def test_every_vision_backend_accepts_a_system_prompt() -> None:
    from familiar_agent.backends.anthropic import AnthropicBackend
    from familiar_agent.backends.gemini import GeminiBackend
    from familiar_agent.backends.openai_compat import OpenAICompatibleBackend

    for cls in (AnthropicBackend, GeminiBackend, OpenAICompatibleBackend):
        params = inspect.signature(cls.complete_with_image).parameters
        assert "system" in params, cls.__name__
