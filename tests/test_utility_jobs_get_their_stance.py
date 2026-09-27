"""軽量LLM の仕事へ、立ち位置と文脈を配る（出-e-は）。

**感情を作るのはパジュである。** 一言要約は一人称で立ち（PAD 評価と相手の気分の分類は出-au 段 5-6 で Jev へ移した）、
発話前の検査と同一意図の判定は外から測る。実測では、パジュとしての立ち位置で発話前の検査を
させると違反18件中0〜1件しか捕まえない（`根拠台帳` §25.8）。**自分で自分は検査できない。**

部品は正本から取る。人格とできることは `capability_state.load_summary()`、家族は
`FAMILY.md`、規則は `loop.prompt.rules_section()`。**手で写した控えを持たない。**

材料が欠けたときは**立ち位置を渡さずに続ける**（いままでと同じ挙動）。`FAMILY.md` が無い
機体や、自己認識をまだ生成していない初回起動でターンを落とさない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.core.context_parts import Stance
from familiar_agent.loop.evaluator import Evaluator


def _backend(reply="0.8 0.1 0.6"):
    be = MagicMock()
    be.complete = AsyncMock(return_value=reply)
    return be


def _evaluator(be, *, context=None):
    return Evaluator(be, MagicMock(), context=context)


# ── 立ち位置が届く ──────────────────────────────────────────────────────────


# 感情の評価と気分の見立ては出-au 段 5-6 で Jev へ移した（Jev にはシステム文が無い・`test_emotion_and_mood_with_jev.py`）。


def test_the_one_line_summary_speaks_as_paju():
    be = _backend("うれしかった")
    seen = {}

    def ctx(stance, *, with_rules=False):
        seen["sum"] = (stance, with_rules)
        return "＜パジュ＞"

    asyncio.run(_evaluator(be, context=ctx).summarize_exchange("やあ", "こんにちは"))
    assert seen["sum"] == (Stance.PAJU, False)


# ── 材料が欠けても落ちない ──────────────────────────────────────────────────


def test_a_missing_part_falls_back_to_no_stance():
    """`FAMILY.md` が無い機体でターンを落とさない。いままでと同じ挙動へ落ちる。"""
    be = _backend("うれしかった")

    def ctx(stance, *, with_rules=False):
        return None

    asyncio.run(_evaluator(be, context=ctx).summarize_exchange("やあ", "こんにちは"))
    assert be.complete.await_args.kwargs["system"] is None


def test_without_a_context_provider_nothing_changes():
    """`context` を渡さなければ、いままでと同じ（システム文なし）。"""
    be = _backend("うれしかった")
    asyncio.run(_evaluator(be).summarize_exchange("やあ", "こんにちは"))
    assert be.complete.await_args.kwargs.get("system") is None


# ── 残り2つ：欲求の列挙（パジュ）と 同一意図の判定（計器）──────────────────


def test_the_same_intent_check_measures_from_outside():
    """語の比較で、感情も人格も関わらない。人格を渡す理由が無い。"""
    from familiar_agent.tools.deferred_search import DeferredSearchTool

    be = _backend("yes")
    seen = {}

    def ctx(stance, *, with_rules=False):
        seen["intent"] = (stance, with_rules)
        return "＜計器＞"

    tool = DeferredSearchTool(AsyncMock(), utility_backend=be, context=ctx)
    assert asyncio.run(tool._is_same_intent("東京の天気", "明日の東京の天気")) is True
    assert seen["intent"] == (Stance.INSTRUMENT, False)
    assert be.complete.await_args.kwargs["system"] == "＜計器＞"


def test_the_same_intent_check_works_without_a_context_provider():
    """提供者が無ければ立ち位置を渡さない（いままでと同じ）。"""
    from familiar_agent.tools.deferred_search import DeferredSearchTool

    be = _backend("yes")
    tool = DeferredSearchTool(AsyncMock(), utility_backend=be)
    assert asyncio.run(tool._is_same_intent("あ", "い")) is True
    assert be.complete.await_args.kwargs.get("system") is None
