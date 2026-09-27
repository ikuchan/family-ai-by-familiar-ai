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


def _backend(reply: str):
    b = AsyncMock()
    b.complete = AsyncMock(return_value=reply)
    return b


# ── ① 相手の気分：読めなければ語ベースの判定へ落ちる ──────────────────────


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


def test_the_satisfied_axes_are_read_through_the_gate() -> None:
    from familiar_agent.core.structured_ask import ask_subset

    axes = {"seeking", "rest", "bond", "safety", "esteem"}
    got = asyncio.run(ask_subset(_backend("seeking, bond"), "p", choices=axes))
    assert got == frozenset({"seeking", "bond"})


# ── ⑤ 場面の JSON：コードフェンス付きでも読める ───────────────────────────


def test_the_scene_reads_json_wrapped_in_a_fence() -> None:
    import familiar_agent.scene as scene

    backend = MagicMock()
    backend.complete = AsyncMock(return_value='```json\n{"entities": [{"label": "cat"}]}\n```')
    del backend.complete_with_image  # 画像なしの経路を通す
    # 引数は (description, backend) の順。
    got = asyncio.run(scene.extract_entities("猫がいる", backend))
    assert got == [{"label": "cat"}]
