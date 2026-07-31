"""ToM ツールの撤去と、一人称 CoT（W 消費型）への置き換えの確認。

三人称の視点分析ツール（tom）を撤去し、応答前に「W の文脈に出る人を一人称で想像
してから一人称で答える」原則をシステムプロンプトに置く。実際の一人称挙動は実機確認、
ここでは撤去と構造（プロンプト原則・在席注入の整形）を見る。
"""

from __future__ import annotations

import importlib

import pytest


def test_tom_module_is_deleted():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("familiar_agent.tools.tom")


def test_event_prompt_has_first_person_cot_and_no_tom():
    """見張る対象は実行中のプロンプト。

    旧 ReAct の `SYSTEM_PROMPT` は撤去した（組み立てる `_system_prompt` に生きた
    呼び出し元が無かった）。同じ性質は `EVENT_SYSTEM_PROMPT` が引き継いでいる。
    """
    from familiar_agent.loop.prompt import EVENT_SYSTEM_PROMPT

    assert "first-person-perspective-taking" in EVENT_SYSTEM_PROMPT
    # 旧 ToM の名残（三人称の視点分析ツール）が消えていること
    assert "theory-of-mind" not in EVENT_SYSTEM_PROMPT
    assert "Theory of Mind" not in EVENT_SYSTEM_PROMPT


def test_format_present_ctx():
    from familiar_agent.agent import format_present_ctx

    assert (
        format_present_ctx("パパ", ["ママ", "太郎"])
        == '(present :speaker "パパ" :others "ママ" "太郎")'
    )
    assert format_present_ctx("パパ", []) == '(present :speaker "パパ")'
    # 話者不明でも壊れない
    assert format_present_ctx("", []) == '(present :speaker "unknown")'
