"""軽量LLM の仕事へ、立ち位置と文脈を配る（出-e-は）。

**感情を作るのはパジュである。** PAD 評価・相手の気分の分類・一言要約は一人称で立ち、
整合チェックと同一意図の判定は外から測る。実測では、パジュとしての立ち位置で整合チェックを
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
from familiar_agent.mood_register import MoodPAD


def _backend(reply="0.8 0.1 0.6"):
    be = MagicMock()
    be.complete = AsyncMock(return_value=reply)
    return be


def _evaluator(be, *, context=None):
    return Evaluator(be, MagicMock(), context=context)


# ── 立ち位置が届く ──────────────────────────────────────────────────────────

def test_the_pad_reading_speaks_as_paju():
    be = _backend()
    seen = {}

    def ctx(stance, *, with_rules=False):
        seen["pad"] = (stance, with_rules)
        return "＜パジュとしての文脈＞"

    asyncio.run(_evaluator(be, context=ctx).emotion_for_turn("やった！", 0.9, mood=MoodPAD()))
    assert seen["pad"] == (Stance.PAJU, False)
    assert be.complete.await_args.kwargs["system"] == "＜パジュとしての文脈＞"


def test_the_companion_mood_reading_speaks_as_paju():
    be = _backend("happy")
    seen = {}

    def ctx(stance, *, with_rules=False):
        seen["mood"] = (stance, with_rules)
        return "＜パジュ＞"

    asyncio.run(_evaluator(be, context=ctx).infer_companion_mood("やった！"))
    assert seen["mood"] == (Stance.PAJU, False)


def test_the_coherence_check_measures_from_outside_and_needs_the_rules():
    """自分で自分は検査できない。規則はシステム文で受け取る。"""
    be = _backend("OK")
    seen = {}

    def ctx(stance, *, with_rules=False):
        seen["coh"] = (stance, with_rules)
        return "＜計器＋規則＞"

    ev = _evaluator(be, context=ctx)
    asyncio.run(ev.check_response_coherence("はい", [{"role": "user", "content": "やあ"}]))
    assert seen["coh"] == (Stance.INSTRUMENT, True)
    assert be.complete.await_args.kwargs["system"] == "＜計器＋規則＞"


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
    be = _backend()

    def ctx(stance, *, with_rules=False):
        return None

    asyncio.run(_evaluator(be, context=ctx).emotion_for_turn("やった！", 0.9, mood=MoodPAD()))
    assert be.complete.await_args.kwargs["system"] is None


def test_without_a_context_provider_nothing_changes():
    """`context` を渡さなければ、いままでと同じ（システム文なし）。"""
    be = _backend()
    asyncio.run(_evaluator(be).emotion_for_turn("やった！", 0.9, mood=MoodPAD()))
    assert be.complete.await_args.kwargs.get("system") is None
