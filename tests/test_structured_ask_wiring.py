"""形のある答えを求める7箇所が、口を通ること（出-d-ろ）。

**落とし先を変えるのは2箇所だけである。** 残る5箇所は、いまの落とし先が既に正しい
（読めなかったことを記録してから倒している）。

| 場所 | いま | 変更後 | 理由 |
|---|---|---|---|
| 相手の気分 | `"engaged"` と断定 | 語ベースの判定へ | 同じファイルに代替がある。「読めなかったから既定」より根拠がある |
| 同じ意図か | 暗黙に「いいえ」 | 文字列の一致へ | 例外時の落とし先と揃える |
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.mood_register import MoodPAD


def _backend(reply: str):
    b = AsyncMock()
    b.complete = AsyncMock(return_value=reply)
    return b


# ── ① 相手の気分：読めなければ語ベースの判定へ落ちる ──────────────────────

def test_an_unreadable_mood_falls_back_to_the_heuristic() -> None:
    """**黙って `"engaged"` と断定しない。** 同じファイルにある語ベースの判定を使う。"""
    from familiar_agent.loop.evaluator import Evaluator, _companion_mood_heuristic

    text = "疲れた。もう寝る"
    ev = Evaluator(utility_backend=_backend("よくわかりません"), backend=object())
    got = asyncio.run(ev.infer_companion_mood(text))
    assert got == _companion_mood_heuristic(text), "語ベースの判定へ落ちていない"


def test_a_readable_mood_is_used() -> None:
    from familiar_agent.loop.evaluator import Evaluator

    ev = Evaluator(utility_backend=_backend("tired"), backend=object())
    assert asyncio.run(ev.infer_companion_mood("何か話す")) == "tired"


# ── ② 同じ意図か：読めなければ文字列の一致へ落ちる ────────────────────────

def _searcher(reply: str):
    from familiar_agent.tools.deferred_search import DeferredSearchTool

    tool = DeferredSearchTool.__new__(DeferredSearchTool)
    tool._utility_backend = _backend(reply)
    return tool


def test_yes_in_japanese_counts_as_the_same_intent() -> None:
    """`.startswith("yes")` は「はい、同じです」を**黙って偽**にしていた。

    偽になると同じ調査を二重に投げる（`求めの版チェーン`「同じ語は二度と調べない」）。
    """
    tool = _searcher("はい、同じです。")
    assert asyncio.run(tool._is_same_intent("天気", "今日の天気")) is True


def test_an_unreadable_intent_falls_back_to_string_equality() -> None:
    """読めなければ文字列の一致で決める（**例外時の落とし先と同じ**）。"""
    tool = _searcher("わかりません")
    assert asyncio.run(tool._is_same_intent("天気", "今日の天気")) is False
    assert asyncio.run(tool._is_same_intent("天気", "天気")) is True


# ── ③ PAD の数値：口を通しても 050 の約束は変わらない ─────────────────────

def test_the_pad_still_goes_unmeasured_when_the_numbers_are_short() -> None:
    from familiar_agent.loop.evaluator import _evaluate_emotion_pad

    pad, a = asyncio.run(_evaluate_emotion_pad(
        _backend("0.7 0.2"), "text", MoodPAD(0.6, 0.3, 0.5, 0.55), arousal=0.8))
    assert pad is None
    assert a == 0.8


# ── ④ 満たされた軸：口を通す ───────────────────────────────────────────────

def test_the_satisfied_axes_are_read_through_the_gate() -> None:
    from familiar_agent.core.structured_ask import ask_subset

    axes = {"seeking", "rest", "bond", "safety", "esteem"}
    got = asyncio.run(ask_subset(_backend("seeking, bond"), "p", choices=axes))
    assert got == frozenset({"seeking", "bond"})


# ── ⑤ 場面の JSON：コードフェンス付きでも読める ───────────────────────────

def test_the_scene_reads_json_wrapped_in_a_fence() -> None:
    import familiar_agent.scene as scene

    backend = MagicMock()
    backend.complete = AsyncMock(
        return_value='```json\n{"entities": [{"label": "cat"}]}\n```')
    del backend.complete_with_image      # 画像なしの経路を通す
    # 引数は (description, backend) の順。
    got = asyncio.run(scene.extract_entities("猫がいる", backend))
    assert got == [{"label": "cat"}]
