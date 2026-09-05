"""キャッシュの寿命を持つ口（出-i）。

§27 で主LLM を `claude-haiku-4-5` ＋キャッシュに確定したが、**その構成はまだ実物に入って
いない**。測定では手で `cache_control` を付けた。入れなければ 738円 のままで、確定した
366円 にならない。

**この API にセッションという保持物は無い。** 毎回1発言だけを送っており、会話履歴はサーバに
残らない。キャッシュは「送った内容の前方一致」に紐づき、会話の継続とは無関係である。

**既定は「何もしない」。** `cli` はキャッシュを持たず、`kimi`／`glm`／`openai_compat` も
持たない。**持たないことが普通である面を必須にしない**（`llm_protocol.py` の既存の判断）。
中身を持つのは `anthropic` だけである（Gemini の明示キャッシュは 出-g で取り下げた——
安定部 1,012 トークンが最小長 1,024 に届かず、作れても保存料のほうが高い）。
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


def test_the_promise_carries_the_three_lifetime_mouths():
    for name in ("warm", "forget", "aclose"):
        assert hasattr(LLMBackend, name), name


@pytest.mark.parametrize("cls", _ALL, ids=[c.__name__ for c in _ALL])
def test_every_backend_answers_the_three(cls):
    for name in ("warm", "forget", "aclose"):
        f = getattr(cls, name, None)
        assert f is not None, f"{cls.__name__}: {name}"
        assert inspect.iscoroutinefunction(f), f"{cls.__name__}: {name}"


@pytest.mark.parametrize(
    "cls", [c for c in _ALL if c is not AnthropicBackend],
    ids=[c.__name__ for c in _ALL if c is not AnthropicBackend])
def test_a_backend_without_a_cache_does_nothing(cls):
    """持たないものは黙って何もしない。**例外を投げない**（呼ぶ側が種類を見分けずに済む）。"""
    be = object.__new__(cls)
    asyncio.run(be.warm("paju", "＜安定部＞"))
    asyncio.run(be.forget("paju"))
    asyncio.run(be.aclose())


# ── Anthropic だけが中身を持つ ──────────────────────────────────────────────

def _anthropic_with_fake():
    seen: dict = {}

    class _Messages:
        async def create(self, **kw):
            seen.update(kw)
            return SimpleNamespace(
                usage=SimpleNamespace(cache_creation_input_tokens=5348,
                                      cache_read_input_tokens=0),
                content=[])

    be = AnthropicBackend(api_key="dummy", model="claude-haiku-4-5-20251001")
    be.client = SimpleNamespace(messages=_Messages())  # type: ignore[assignment]
    return be, seen


def test_warming_sends_the_stable_part_with_cache_control():
    be, seen = _anthropic_with_fake()
    asyncio.run(be.warm("paju", "＜安定部＞"))
    blocks = seen["system"]
    assert blocks[0]["text"] == "＜安定部＞"
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert seen["max_tokens"] == 1          # 1トークンで足りる（載せるだけ）


def test_warming_remembers_which_keys_are_alive():
    be, _ = _anthropic_with_fake()
    asyncio.run(be.warm("paju", "＜A＞"))
    asyncio.run(be.warm("instrument", "＜B＞"))
    assert be.warm_keys() == {"paju", "instrument"}


def test_forgetting_drops_the_key_without_calling_the_api():
    """`ephemeral` は保存料が無く5分で自然に消えるので、API を呼ぶ必要がない。"""
    be, seen = _anthropic_with_fake()
    asyncio.run(be.warm("paju", "＜A＞"))
    seen.clear()
    asyncio.run(be.forget("paju"))
    assert be.warm_keys() == set()
    assert seen == {}


def test_closing_drops_every_key():
    be, _ = _anthropic_with_fake()
    asyncio.run(be.warm("paju", "＜A＞"))
    asyncio.run(be.warm("instrument", "＜B＞"))
    asyncio.run(be.aclose())
    assert be.warm_keys() == set()


def test_warming_the_same_key_again_replaces_the_stable_part():
    """自己認識が更新されたら安定部が変わる。古いものを持ち続けない。"""
    be, seen = _anthropic_with_fake()
    asyncio.run(be.warm("paju", "＜古い＞"))
    asyncio.run(be.warm("paju", "＜新しい＞"))
    assert seen["system"][0]["text"] == "＜新しい＞"
    assert be.warm_keys() == {"paju"}


def test_a_failure_to_warm_is_not_fatal():
    """温められなくてもターンは回る。キャッシュは速さと安さのためのもので、機能ではない。"""
    class _Messages:
        async def create(self, **kw):
            raise RuntimeError("529 overloaded")

    be = AnthropicBackend(api_key="dummy", model="m")
    be.client = SimpleNamespace(messages=_Messages())  # type: ignore[assignment]
    asyncio.run(be.warm("paju", "＜安定部＞"))
    assert be.warm_keys() == set()          # 載らなかったので覚えない
