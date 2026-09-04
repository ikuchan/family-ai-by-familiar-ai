"""`complete()` にシステム文を渡す口（出-e）。

軽量LLM には、パジュが誰か・どんな道具を持ち・どんな決まりを守っているかが**一切渡って
いなかった**。整合チェックは規則との照合が仕事なのに、規則は主LLM のシステムプロンプトに
しかない。実測では、渡さないと違反18件中3件しか捕まえず、渡すと18件中18件を捕まえた
（`根拠台帳` §25.8）。

**native な口で渡す。** プロンプトの先頭へ足すのとは別物で、モデルはシステム文と利用者の
文を違う重みで扱う。5つは native な口を持ち、`cli` だけ持たないので前置きで代替する。

キーワード引数・既定 `None` なので、いまの呼び出しは1つも壊れない。
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from familiar_agent.backends import (
    AnthropicBackend,
    CLIBackend,
    GeminiBackend,
    GLMBackend,
    KimiBackend,
    OpenAICompatibleBackend,
)
from familiar_agent.core.llm_protocol import LLMBackend

_ALL = (AnthropicBackend, CLIBackend, GeminiBackend, GLMBackend,
        KimiBackend, OpenAICompatibleBackend)


def test_the_promise_carries_a_system_text():
    sig = inspect.signature(LLMBackend.complete)
    assert "system" in sig.parameters
    p = sig.parameters["system"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY
    assert p.default is None


@pytest.mark.parametrize("cls", _ALL, ids=[c.__name__ for c in _ALL])
def test_every_backend_takes_it_the_same_way(cls):
    sig = inspect.signature(cls.complete)
    assert "system" in sig.parameters, cls.__name__
    p = sig.parameters["system"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY, cls.__name__
    assert p.default is None, cls.__name__


# ── 実際に届くか（偽のクライアントで確かめる）────────────────────────────────

def test_gemini_sends_it_as_a_system_instruction():
    seen = {}

    class _Models:
        async def generate_content(self, *, model, contents, config):
            seen["system"] = config.system_instruction
            return SimpleNamespace(text="ok")

    be = GeminiBackend(api_key="dummy", model="m")
    be._client = SimpleNamespace(aio=SimpleNamespace(models=_Models()))  # type: ignore[assignment]
    be._retry_base = 0.0
    assert asyncio.run(be.complete("問い", 10, system="立ち位置")) == "ok"
    assert seen["system"] == "立ち位置"


def test_the_openai_shaped_backends_send_it_as_a_system_message():
    for cls in (OpenAICompatibleBackend, KimiBackend, GLMBackend):
        seen = {}

        class _Completions:
            async def create(self, **kw):
                seen["messages"] = kw["messages"]
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

        be = object.__new__(cls)
        be.client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
        be.model = "m"
        be._use_completion_tokens = False
        assert asyncio.run(be.complete("問い", 10, system="立ち位置")) == "ok", cls.__name__
        assert seen["messages"][0] == {"role": "system", "content": "立ち位置"}, cls.__name__
        assert seen["messages"][1]["content"] == "問い", cls.__name__


def test_the_cli_backend_puts_it_in_front_of_the_prompt():
    """native な口が無いので前置きで代替する。ここだけ他と揃わない。"""
    seen = {}

    async def _run(prompt):
        seen["prompt"] = prompt
        return "ok"

    be = object.__new__(CLIBackend)
    be._run = _run  # type: ignore[method-assign]
    assert asyncio.run(be.complete("問い", 10, system="立ち位置")) == "ok"
    assert seen["prompt"].startswith("立ち位置")
    assert seen["prompt"].endswith("問い")


def test_leaving_it_out_changes_nothing():
    """既定 `None` なら、いままでと同じものが飛ぶ。"""
    seen = {}

    class _Models:
        async def generate_content(self, *, model, contents, config):
            seen["system"] = config.system_instruction
            return SimpleNamespace(text="ok")

    be = GeminiBackend(api_key="dummy", model="m")
    be._client = SimpleNamespace(aio=SimpleNamespace(models=_Models()))  # type: ignore[assignment]
    be._retry_base = 0.0
    asyncio.run(be.complete("問い", 10))
    assert seen["system"] is None
