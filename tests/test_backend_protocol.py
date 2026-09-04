"""バックエンドの約束を型で表す（出-d-は）。

**6つのバックエンドは、既に同じ5つのメソッドを同じ署名で持っていた**（実測）。だが
抽象基底も `Protocol` も無く、**暗黙の約束で並んでいた**——新しいモデルを足すとき、
何を実装すれば足りるのかがコードから読めない。

**揃っていないものは任意として表す。** `complete_with_image` は3つ（Anthropic・
OpenAICompatible・Gemini）、`make_system_message` は2つ（Kimi・GLM）しか持たない。
持つかどうかを呼ぶ側が `hasattr` で見ている箇所があるので、**型にもそう出す**。
"""

from __future__ import annotations

import inspect

import pytest

from familiar_agent.backend import (
    AnthropicBackend,
    CLIBackend,
    GeminiBackend,
    GLMBackend,
    KimiBackend,
    OpenAICompatibleBackend,
)
from familiar_agent.core.llm_protocol import LLMBackend

_ALL = [
    AnthropicBackend, OpenAICompatibleBackend, KimiBackend,
    GLMBackend, GeminiBackend, CLIBackend,
]


@pytest.mark.parametrize("cls", _ALL, ids=lambda c: c.__name__)
def test_every_backend_satisfies_the_protocol(cls) -> None:
    """**6つとも約束を満たす。** 満たさないものが混ざれば、ここで落ちる。"""
    assert issubclass(cls, LLMBackend), f"{cls.__name__} が約束を満たしていない"


@pytest.mark.parametrize("cls", _ALL, ids=lambda c: c.__name__)
def test_the_signatures_match_the_protocol(cls) -> None:
    """名前だけでなく**引数まで**揃っていること。

    `Protocol` の `issubclass` は引数を見ないので（実行時は名前だけ）、ここで見る。
    """
    for name in ("stream_turn", "complete", "make_user_message",
                 "make_assistant_message", "make_tool_results"):
        want = list(inspect.signature(getattr(LLMBackend, name)).parameters)
        got = list(inspect.signature(getattr(cls, name)).parameters)
        assert got == want, f"{cls.__name__}.{name} の引数が違う: {got} ≠ {want}"


def test_the_optional_face_is_documented() -> None:
    """任意のものは約束に**含めない**（持たないバックエンドがあるため）。

    呼ぶ側は `hasattr` で見る（`scene.py` が `complete_with_image` をそうしている）。
    """
    names = set(dir(LLMBackend))
    assert "complete_with_image" not in names, "任意のものを必須にしている"
    assert "make_system_message" not in names
