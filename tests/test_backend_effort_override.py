"""思考の深さ（effort）を呼び出しごとに渡せるようにする。

段階2 で軽量LLM が「どれくらい深く考えるべきか」を決めるため、生成のたびに effort を
指定できる必要がある。既定（None）は従来どおりインスタンスの設定を使う。
"""

from __future__ import annotations

from familiar_agent.backends import AnthropicBackend


def _backend(effort: str = "high") -> AnthropicBackend:
    b = AnthropicBackend.__new__(AnthropicBackend)
    b.model = "claude-sonnet-4-6"
    b.thinking_mode = "auto"
    b.thinking_budget = 10000
    b.thinking_effort = effort
    return b


def test_call_level_effort_overrides_the_instance_setting():
    params = _backend("high")._build_thinking_params(effort="low")
    assert params["output_config"]["effort"] == "low"


def test_default_keeps_the_instance_setting():
    # high でも effort を送る（既定はモデルで違う・2026-11-14 本人承認）。
    assert _backend("high")._build_thinking_params()["output_config"]["effort"] == "high"
    assert _backend("medium")._build_thinking_params()["output_config"]["effort"] == "medium"
