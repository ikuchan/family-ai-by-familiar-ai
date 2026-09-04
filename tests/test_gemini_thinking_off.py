"""Gemini に「思考は要らない」と伝える言い方を、モデルごとに探って覚える。

思考を切る言い方は、モデルによって通るものが違う（2026-09-04 実測）。決め打ちで
1通りしか送らないと、受け付けないモデルでは毎回 400 になり、無視するモデルでは思考が
`max_tokens` を食い切って空が返る。空は `structured_ask` から見ると「形を外した」なので、
PAD は未測定になり、気分の分類は語ベースへ落ちる。**静かに品質が下がる。**

ここでは実 API を叩かない。偽のクライアントを差し込んで、何を送ったかで確かめる。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from familiar_agent.backends import GeminiBackend

_ANSWER = "0.9 0.8 0.7"


def _form_of(thinking) -> str:
    """送られた思考設定が3通りのどれかを言い当てる。"""
    if thinking is None:
        return "none"
    if getattr(thinking, "thinking_budget", None) == 0:
        return "budget"
    if getattr(thinking, "thinking_level", None):
        return "level"
    return "other"


def _chunk(text: str):
    part = SimpleNamespace(text=text, thought=False, function_call=None)
    content = SimpleNamespace(parts=[part])
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)])


class _FakeModels:
    """`reject` に入れた言い方だけを 400 で拒む偽のモデル群。"""

    def __init__(self, reject: set[str]) -> None:
        self.reject = reject
        self.seen: list[str] = []

    def _guard(self, config) -> None:
        form = _form_of(config.thinking_config)
        self.seen.append(form)
        if form in self.reject:
            raise RuntimeError("400 INVALID_ARGUMENT. Request contains an invalid argument.")

    async def generate_content(self, *, model, contents, config):
        self._guard(config)
        return SimpleNamespace(text=_ANSWER)

    async def generate_content_stream(self, *, model, contents, config):
        self._guard(config)

        async def _gen():
            yield _chunk(_ANSWER)

        return _gen()


def _backend(reject: set[str]) -> tuple[GeminiBackend, _FakeModels]:
    be = GeminiBackend(api_key="dummy-key-for-test", model="gemini-test")
    fake = _FakeModels(reject)
    be._client = SimpleNamespace(aio=SimpleNamespace(models=fake))  # type: ignore[assignment]
    be._retry_base = 0.0
    return be, fake


# ── 成り立つ側（実装の前後どちらでも通るべきもの）────────────────────────────

def test_budget_zero_is_tried_first():
    """いま動いている gemini-2.5-flash の送り方を変えない。"""
    be, fake = _backend(reject=set())
    out = asyncio.run(be.complete("なにか", 20))
    assert out == _ANSWER
    assert fake.seen == ["budget"]


def test_complete_returns_empty_when_every_form_is_rejected():
    """どの言い方も拒まれたら、例外を投げずに空を返す（従来どおり）。"""
    be, fake = _backend(reject={"budget", "level", "none"})
    assert asyncio.run(be.complete("なにか", 20)) == ""


# ── 間違っていれば見えるはずの側 ────────────────────────────────────────────

def test_level_is_used_when_budget_is_rejected():
    """3.5-flash-lite は budget=0 を 400 で拒み、level="low" なら通る。"""
    be, fake = _backend(reject={"budget"})
    out = asyncio.run(be.complete("なにか", 20))
    assert out == _ANSWER
    assert fake.seen == ["budget", "level"]


def test_nothing_is_sent_when_both_are_rejected():
    """どちらの言い方も拒むモデルでは、思考設定を送らずに聞く。"""
    be, fake = _backend(reject={"budget", "level"})
    out = asyncio.run(be.complete("なにか", 20))
    assert out == _ANSWER
    assert fake.seen == ["budget", "level", "none"]


def test_the_working_form_is_remembered():
    """一度通った言い方は覚える。毎回3通り試して遅くならない。"""
    be, fake = _backend(reject={"budget"})
    asyncio.run(be.complete("ひとつめ", 20))
    asyncio.run(be.complete("ふたつめ", 20))
    assert fake.seen == ["budget", "level", "level"]


def test_stream_turn_negotiates_too():
    """対話ターンも同じ交渉を通る。ここが直らないと主LLM ごと 400 になる。"""
    be, fake = _backend(reject={"budget"})
    result, _raw = asyncio.run(
        be.stream_turn("システム", [{"role": "user", "content": "やあ"}], [], 64, None)
    )
    assert result.text == _ANSWER
    assert fake.seen == ["budget", "level"]
