"""どの担い手も「道具を使った発言」を一から組める（出-aq 段 2・2026-09-25）。

つなぎを主LLM の会話に置くには、**「つなぎの道具を使った」という発言**を作る必要がある。
道具の返り（`make_tool_results`）は 6 つの担い手すべてに共通の口があったが、道具を使った
側の発言（`make_assistant_message`）は API の生の返りをそのまま使う作りで、**一から組む
口が無かった**。

いまの主LLM は Anthropic だが、**6 つすべてに足す**（本人「全部に足して」）。バックエンドは
特定の実装に依存しない（`CLAUDE.md`）。形は担い手ごとに 4 通りある：

| 担い手 | 形 |
|---|---|
| Anthropic | `content` に `tool_use` の塊 |
| Gemini | `role: model` の `parts` に `function_call` |
| OpenAI 互換（道具を API で渡す）・Kimi・GLM | `tool_calls` に `type: function` |
| OpenAI 互換（道具を文で書かせる）・CLI | 本文に `<tool_call>{…}</tool_call>` |

返りと往復で組めること（同じ `ToolCall` の id で対になる）も確かめる。
"""

from __future__ import annotations

import json

from familiar_agent.backends.types import ToolCall

CALL = ToolCall(
    id="call_filler01", name="filler", input={"text": "こんにちは！ちょっと待ってくださいね。"}
)


def _anthropic():
    from familiar_agent.backends.anthropic import AnthropicBackend

    return AnthropicBackend.__new__(AnthropicBackend)


def _gemini():
    from familiar_agent.backends.gemini import GeminiBackend

    return GeminiBackend.__new__(GeminiBackend)


def _openai(mode: str):
    from familiar_agent.backends.openai_compat import OpenAICompatibleBackend

    b = OpenAICompatibleBackend.__new__(OpenAICompatibleBackend)
    b.tools_mode = mode
    return b


def _kimi():
    from familiar_agent.backends.kimi import KimiBackend

    return KimiBackend.__new__(KimiBackend)


def _glm():
    from familiar_agent.backends.glm import GLMBackend

    return GLMBackend.__new__(GLMBackend)


def _cli():
    from familiar_agent.backends.cli import CLIBackend

    return CLIBackend.__new__(CLIBackend)


# ── 形 ───────────────────────────────────────────────────────────────────


def test_anthropic_states_a_tool_use_block():
    msg = _anthropic().make_tool_call_message([CALL])
    assert msg["role"] == "assistant"
    (block,) = msg["content"]
    assert block == {
        "type": "tool_use",
        "id": "call_filler01",
        "name": "filler",
        "input": CALL.input,
    }


def test_gemini_states_a_function_call_part():
    msg = _gemini().make_tool_call_message([CALL])
    assert msg["role"] == "model"
    (part,) = msg["parts"]
    assert part["function_call"] == {"name": "filler", "args": CALL.input}


def test_gemini_signs_the_call_it_made_up():
    """**署名が無いと Gemini 3 系は拒む**（2026-09-25 実測：`400 Function call is missing a
    thought_signature`）。形のテストは通っていたのに、実物は受け付けなかった。

    Google の文書が、作った呼び出しのために検証を飛ばす値を定めている。
    """
    (part,) = _gemini().make_tool_call_message([CALL])["parts"]
    assert part["thought_signature"] == "skip_thought_signature_validator"


def _openai_style(msg):
    assert msg["role"] == "assistant"
    (tc,) = msg["tool_calls"]
    assert tc["id"] == "call_filler01"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "filler"
    assert json.loads(tc["function"]["arguments"]) == CALL.input


def test_openai_native_states_tool_calls():
    _openai_style(_openai("native").make_tool_call_message([CALL]))


def test_kimi_states_tool_calls():
    _openai_style(_kimi().make_tool_call_message([CALL]))


def test_glm_states_tool_calls():
    _openai_style(_glm().make_tool_call_message([CALL]))


def _prompt_style(msg):
    """道具を文で書かせる担い手は、**モデル自身が書くのと同じ形**で書く。"""
    assert msg["role"] == "assistant"
    text = msg["content"]
    assert text.startswith("<tool_call>") and text.endswith("</tool_call>")
    body = json.loads(text[len("<tool_call>") : -len("</tool_call>")])
    assert body == {"name": "filler", "input": CALL.input}


def test_openai_prompt_mode_writes_the_tag():
    _prompt_style(_openai("prompt").make_tool_call_message([CALL]))


def test_cli_writes_the_tag():
    _prompt_style(_cli().make_tool_call_message([CALL]))


def test_the_prompt_style_is_read_back_as_the_same_call():
    """書いた形を、その担い手自身の読み取りが同じ道具として読む。**往復で壊れない。**"""
    from familiar_agent.backends.shared import _parse_tool_calls_from_text

    text = _cli().make_tool_call_message([CALL])["content"]
    (got,) = _parse_tool_calls_from_text(text)
    assert got.name == "filler"
    assert got.input == CALL.input


# ── 複数でも組める ────────────────────────────────────────────────────────


def test_several_calls_go_into_one_message():
    """つなぎは 1 回とは限らない（調べもの待ちでは続けて出る）。"""
    two = [CALL, ToolCall(id="call_filler02", name="filler", input={"text": "もう少しです。"})]
    assert len(_anthropic().make_tool_call_message(two)["content"]) == 2
    assert len(_gemini().make_tool_call_message(two)["parts"]) == 2
    assert len(_openai("native").make_tool_call_message(two)["tool_calls"]) == 2
    assert _cli().make_tool_call_message(two)["content"].count("<tool_call>") == 2


# ── 返りと対になる ────────────────────────────────────────────────────────


def test_the_call_and_its_result_pair_up_by_id():
    """道具の返りは既存の `make_tool_results` で作る。**同じ id で対になる**こと。"""
    b = _anthropic()
    call = b.make_tool_call_message([CALL])
    (result,) = b.make_tool_results([CALL], [("Said: こんにちは！", None)])
    assert call["content"][0]["id"] == result["content"][0]["tool_use_id"]
