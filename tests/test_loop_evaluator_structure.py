"""evaluator の切り出し（loop/evaluator.py）の構造確認。

agent.py から評価器（軽量LLM を使う感情・要約・相手気分・整合性チェック）を
独立モジュールへ分離したことを、振る舞いではなく構造で確かめる少数のテスト。
"""

from __future__ import annotations

import inspect

import pytest

from familiar_agent.mood_register import MoodPAD


def test_evaluator_class_exists_with_injected_backends() -> None:
    """Evaluator は utility_backend と backend を受け取って構築される。"""
    from familiar_agent.loop.evaluator import Evaluator

    ev = Evaluator(utility_backend="U", backend="B")
    assert ev._utility_backend == "U"
    assert ev.backend == "B"


def test_evaluator_exposes_public_methods() -> None:
    """4つの評価メソッドを公開シグネチャで持つ。"""
    from familiar_agent.loop.evaluator import Evaluator

    for name in (
        "emotion_for_turn",
        "summarize_exchange",
        "infer_companion_mood",
        "check_response_coherence",
    ):
        assert callable(getattr(Evaluator, name)), name


def test_module_helpers_moved_out_of_agent() -> None:
    """_evaluate_emotion_pad / A_GATE / 各プロンプトは evaluator 側にある。"""
    from familiar_agent.loop import evaluator as ev

    assert hasattr(ev, "_evaluate_emotion_pad")
    assert hasattr(ev, "A_GATE")
    assert hasattr(ev, "_companion_mood_heuristic")


def test_agent_does_not_redefine_evaluator_bodies() -> None:
    """agent.py に評価器の本体（プロンプト定義・PAD 評価関数）が残っていない。"""
    import familiar_agent.agent as agent_mod

    src = inspect.getsource(agent_mod)
    assert "_EMOTION_PAD_PROMPT = " not in src, "PAD プロンプトが agent.py に残存"
    assert "_COMPANION_MOOD_PROMPT = " not in src, "相手気分プロンプトが agent.py に残存"
    assert "async def _evaluate_emotion_pad" not in src, "PAD 評価関数が agent.py に残存"


@pytest.mark.asyncio
async def test_emotion_for_turn_returns_pad_and_label() -> None:
    """emotion_for_turn は (MoodPAD, ラベル) を返す（A_gate 未満は評価器を呼ばない）。"""
    from familiar_agent.loop.evaluator import Evaluator

    ev = Evaluator(utility_backend=object(), backend=object())
    pad, label = await ev.emotion_for_turn("淡々とした文", arousal=0.0)
    assert isinstance(pad, MoodPAD)
    assert isinstance(label, str) and label
