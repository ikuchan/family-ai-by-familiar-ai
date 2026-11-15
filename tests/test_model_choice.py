"""主LLM と軽量LLM の選び分け（2026-11-14・本人承認の実装計画「主LLM を Opus 5.5 の fast mode、軽量LLM を Haiku 5.5 にし、
選び分けを config.py へ移す」）。

モデルと考え方は `config.py` の既定値で決め、`.env` からは読まない（`.env` は資格情報と最初の起動に要るものだけ・開発ルール）。
速さの測り（作った問い 10 問・本物の道具 18 個）：会話の 1 回の中央は Opus 5.5 fast 0.85 秒、Sonnet 5.5 1.51〜1.60 秒、
ふつうの Opus 5.5 1.65 秒。軽量の書き手は Haiku 5.5 0.97 秒で、10 回とも素の JSON。
"""

from __future__ import annotations

import pytest

from familiar_agent.config import AgentConfig

_OLD = {
    "PLATFORM": "gemini",
    "MODEL": "old-model",
    "UTILITY_PLATFORM": "gemini",
    "UTILITY_MODEL": "old-light",
    "THINKING_MODE": "disabled",
    "THINKING_EFFORT": "high",
}


@pytest.fixture
def old_env(monkeypatch):
    for k, v in _OLD.items():
        monkeypatch.setenv(k, v)


def test_the_models_and_thinking_come_from_config_not_env(old_env):
    """`.env` に古い値が残っていても効かない（本人が行を消し忘れても既定値が効く）。"""
    cfg = AgentConfig()
    assert (cfg.platform, cfg.model) == ("anthropic", "claude-opus-5-5")
    assert (cfg.utility_platform, cfg.utility_model) == ("anthropic", "claude-haiku-5-5")
    assert (cfg.thinking_mode, cfg.thinking_effort) == ("adaptive", "low")


def test_the_fast_mode_and_its_fallback_are_config_values():
    cfg = AgentConfig()
    assert cfg.main_llm_fast is True
    has_fallback = hasattr(
        cfg, "main_llm_fallback_model"
    )  # 落ちたときに設定を丸ごと表示しない（鍵が出る）
    assert not has_fallback, "429 はモデルを替えず、ふつうの速さで送り直す（本人「ウ」）"
    assert cfg.opus_complete_effort == "medium"
    assert cfg.opus_complete_extra_tokens == 2000


def test_the_light_backend_is_haiku_on_the_anthropic_key(monkeypatch):
    """軽量LLM は主LLM と同じ Anthropic の鍵で作る。`UTILITY_API_KEY` は声の合成（Gemini）の鍵なので使わない。"""
    from familiar_agent.backends.anthropic import AnthropicBackend
    from familiar_agent.backends.factory import create_backend, create_utility_backend

    monkeypatch.setenv("API_KEY", "main-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "main-key")
    monkeypatch.setenv("UTILITY_API_KEY", "gemini-tts-key")
    cfg = AgentConfig()
    light = create_utility_backend(cfg)
    assert isinstance(light, AnthropicBackend)
    assert light is not create_backend(cfg)
    model = light.model
    uses_main_key = light.client.api_key == "main-key"  # 落ちたときに鍵の値を表示しない
    assert model == "claude-haiku-5-5"
    assert uses_main_key, "軽量LLM が主LLM の鍵を使っていない"


def test_scene_reading_falls_back_to_the_light_backend():
    """場面の読み取り（写真を読む）は、専用の設定が無ければ軽量LLM（Haiku 5.5）に任せる（agent.py の `or`）。"""
    from familiar_agent.backends.factory import create_scene_backend

    assert create_scene_backend(AgentConfig()) is None


# ── 会話の fast mode と、429 のときのやり直し（作業 5）──────────────────────────────


class _FakeStream:
    """本物の SDK と同じく、`stream()` は管理役を返し、失敗は `async with` に入るとき（`__aenter__`）に上がる。"""

    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        if isinstance(self._response, Exception):
            raise self._response
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def text_stream(self):
        async def gen():
            if False:
                yield ""

        return gen()

    async def get_final_message(self):
        return self._response


def _response(speed: str | None = None):
    from types import SimpleNamespace

    from anthropic.types import TextBlock

    usage = SimpleNamespace(input_tokens=10, output_tokens=5, speed=speed)
    return SimpleNamespace(content=[TextBlock(type="text", text="はい")], usage=usage)


def _rate_limit():
    import anthropic
    import httpx

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.RateLimitError("429", response=httpx.Response(429, request=req), body=None)


class _Messages:
    def __init__(self, outcomes):
        self.calls: list[dict] = []
        self._outcomes = list(outcomes)

    def stream(self, **kw):
        self.calls.append(kw)
        return _FakeStream(self._outcomes.pop(0))


def _fast_backend(outcomes):
    from types import SimpleNamespace

    from familiar_agent.backends.anthropic import AnthropicBackend

    b = AnthropicBackend(
        api_key="x",
        model="claude-opus-5-5",
        thinking_mode="adaptive",
        thinking_effort="low",
        fast=True,
    )
    b.client = SimpleNamespace(messages=_Messages(outcomes))
    return b


def _turn(b):
    import asyncio

    return asyncio.run(b.stream_turn("sys", [{"role": "user", "content": "q"}], [], 1024, None))


def test_a_conversation_turn_asks_for_fast_mode():
    """会話の 1 回は fast mode で頼む。やり直しは SDK の自動のやり直しに任せる（2026-11-15 本人）。"""
    from familiar_agent.backends.factory import create_backend

    b = _fast_backend([_response("fast")])
    _turn(b)
    calls = b.client.messages.calls
    assert len(calls) == 1
    assert calls[0]["extra_body"] == {"speed": "fast"}
    assert calls[0]["model"] == "claude-opus-5-5"
    main = create_backend(AgentConfig())
    assert main.fast is True


def test_a_fast_mode_429_resends_the_same_opus_request_at_standard_speed(caplog):
    """fast mode の 429 では、モデルを替えずに `speed` だけを外し、ふつうの速さで送り直す（2026-11-14 本人「ウ」）。

    思考のブロック（Opus のものは Opus だけが読む）、調停が選んだ effort、道具の定義のキャッシュが保たれる
    （system と会話のキャッシュは `speed` を変えると書き直しになる。API の資料の無効化の表）。
    この 429 は、SDK が自動のやり直しを使い切ったあとのもの（偽の client では 1 回で返す）。
    """
    b = _fast_backend([_rate_limit(), _response()])
    with caplog.at_level("INFO"):
        result, _ = _turn(b)
    assert result.text == "はい"
    fast, resent = b.client.messages.calls
    assert "extra_body" not in resent
    same = {k: v for k, v in fast.items() if k != "extra_body"} == resent
    assert same, "送り直しの中身が fast mode の回と違う"
    lines = [r for r in caplog.records if "fast mode が 429" in r.getMessage()]
    assert [r.levelname for r in lines] == [
        "WARNING"
    ]  # 回復した異常なので WARNING（コード規約・2026-11-15 本人「ア」）


def test_a_fast_mode_429_does_not_wait_but_a_standard_429_still_does(monkeypatch):
    """fast mode の 429 のあとは待たずに送り直す。ふつうの速さの 429 は、いまどおり 60 秒待つ（反証の側）。"""
    from familiar_agent.backends import anthropic as mod

    waits: list[float] = []

    async def fake_sleep(sec):
        waits.append(sec)

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)
    b = _fast_backend([_rate_limit(), _rate_limit(), _response()])
    result, _ = _turn(b)
    assert result.text == "はい"
    assert waits == [60]
    assert len(b.client.messages.calls) == 3


def test_after_the_fast_mode_429_the_standard_retries_stop_at_two(monkeypatch):
    """fast mode の 429 のあと、ふつうの速さの 429 は 60 秒、120 秒と待って 2 回まで。3 回目の 429 は上へ上げる（止まらずに回り続けない）。"""
    import anthropic

    from familiar_agent.backends import anthropic as mod

    waits: list[float] = []

    async def fake_sleep(sec):
        waits.append(sec)

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)
    b = _fast_backend([_rate_limit() for _ in range(4)])
    with pytest.raises(anthropic.RateLimitError):
        _turn(b)
    assert waits == [60, 120]
    assert len(b.client.messages.calls) == 4


def test_the_light_backend_does_not_ask_for_fast_mode(monkeypatch):
    """**反証の側**：fast mode は主LLM の会話だけ。軽量LLM（Haiku）は頼まない。"""
    from familiar_agent.backends.factory import create_utility_backend

    monkeypatch.setenv("API_KEY", "main-key")
    light = create_utility_backend(AgentConfig())
    assert light.fast is False


def test_the_usage_line_says_which_speed_ran(caplog):
    b = _fast_backend([_response("fast")])
    with caplog.at_level("INFO"):
        _turn(b)
    assert any("速さ=fast" in r.getMessage() for r in caplog.records)


def test_the_settings_screen_no_longer_offers_the_model_choices():
    """モデルと考え方は `config.py` の既定値で決めるので、設定の画面からは選ばせない（`.env` へ書かせない）。"""
    from familiar_agent import settings_schema

    keys = {f.env_key for f in settings_schema.SETTINGS_FIELDS}
    gone = {
        "PLATFORM",
        "MODEL",
        "UTILITY_PLATFORM",
        "UTILITY_MODEL",
        "THINKING_MODE",
        "THINKING_EFFORT",
    }
    assert not keys & gone, sorted(keys & gone)
    assert "API_KEY" in keys  # 反証の側：資格情報は画面に残る


def test_opus5_is_told_apart_in_one_place():
    """`backends/anthropic.py` の中で、Opus 5 系かどうかの見分けは 1 か所（`_is_opus5`）にまとめる（2026-11-15 本人「イ」）。"""
    from familiar_agent.backends.anthropic import _is_opus5

    assert _is_opus5("claude-opus-5-5")
    assert not _is_opus5("claude-sonnet-5-5")  # 反証の側
    assert not _is_opus5("claude-haiku-5-5")
