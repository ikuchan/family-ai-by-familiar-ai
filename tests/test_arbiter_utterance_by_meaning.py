"""発話は意味 → 動作を先読みの 1 回で聞く（出-ay 段 4-4b・2026-10-09・`設計方針_判定の段` §2.2.5）。

人の言葉への最初の反復は、`core/utterance_meaning` の問い（意味＋意味ごとの動作）を Jev に 1 回で聞き、決まりで最終の動作を
決めて `Decision` に写す。越えなければ「よく考えるか、軽く聞き返すか」をもう 1 回聞く。黙ると決まったら、発話の最初の反復でも
主LLM を呼ばずに閉じる。黙る依頼・名乗り・否定・時期は段 4-4c で戻す（本人：一時的に効かないのはかまわない）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.loop import iteration
from familiar_agent.loop.arbiter import Decision
from tests._arbiter_fakes import decide, prompt_of, writer_says

_FAMILY = """## ゆうすけ
- **名前**：雄輔
- **呼び方**：パパ、ゆうすけ
## たいき
- **名前**：泰輝
- **呼び方**：たいき
"""


def _jev(*answers: dict) -> MagicMock:
    """呼ばれるたびに順に答える偽の Jev（answers は問いの鍵 → 答え）。"""
    queue = [JevAnswer(ok=True, answers=a) for a in answers]

    async def ask(state, questions):
        return queue.pop(0) if len(queue) > 1 else queue[0]

    jev = MagicMock()
    jev.available = True
    jev.ask = AsyncMock(side_effect=ask)
    return jev


def _a(choice: str, confidence: float) -> dict:
    return {"choice": choice, "confidence": confidence}


def _run(jev, *, writer=None, extra=("play_music", "stop_music")):
    writer = writer or writer_says(
        {"text": "もう一度お願いします", "tool_input": {"name": "ケイマン"}, "query": "明日の天気"}
    )
    d = asyncio.run(
        decide(
            jev=jev,
            writer=writer,
            utterance="パジュ、ケイマンかけて",
            origin="発話",
            can_see=True,
            extra_actions=extra,
            family_md=_FAMILY,
        )
    )
    return d, writer


def test_the_meaning_and_its_actions_are_asked_at_once():
    jev = _jev({"meaning": _a("music", 0.9), "action_music": _a("play_music", 0.9)})
    d, _ = _run(jev)
    (_state, questions), _ = jev.ask.await_args
    assert "meaning" in questions and "action_music" in questions
    assert "branch" not in questions
    assert set(questions["action_claim"]["criteria"]) == {"パパ", "たいき", "other"}
    assert (d.branch, d.action, d.query) == ("action", "play_music", "ケイマン")


def test_unformed_is_silent_without_writing():
    jev = _jev({"meaning": _a("unformed", 0.2)})
    d, writer = _run(jev)
    assert (d.branch, d.text) == ("light", "")
    writer.complete.assert_not_awaited()


def test_ask_back_is_written_by_the_light_llm():
    jev = _jev({"meaning": _a("music", 0.9), "action_music": _a("ask_back", 0.2)})
    d, writer = _run(jev)
    assert (d.branch, d.text) == ("light", "もう一度お願いします")
    assert "聞き返す" in prompt_of(writer)


def test_when_unsure_it_asks_think_or_ask_back():
    jev = _jev({"meaning": _a("research", 0.5)}, {"action": _a("reply_full", 0.7)})
    d, _ = _run(jev)
    assert jev.ask.await_count == 2
    assert set(jev.ask.await_args.args[1]["action"]["criteria"]) == {"reply_full", "ask_back"}
    assert d.branch == "full"
    jev = _jev({"meaning": _a("research", 0.5)}, {"action": _a("reply_full", 0.4)})
    d, _ = _run(jev)
    assert (d.branch, d.text) == ("light", "もう一度お願いします")  # それも越えなければ軽く聞き返す


def test_answering_from_what_is_known():
    jev = _jev({"meaning": _a("answerable", 0.9), "action_answerable": _a("reply_light", 0.8)})
    d, _ = _run(jev)
    assert (d.branch, d.text) == ("light", "もう一度お願いします")


def test_a_quiet_decision_closes_silently_for_a_conversation_too():
    quiet = Decision(branch="light", text="")
    assert iteration.closes_silently(quiet, trigger_kind="発話", returned=frozenset())


def test_the_recalled_memories_have_their_own_heading():
    import inspect

    from familiar_agent.loop import workspace

    assert "[思い出したこと]" in inspect.getsource(workspace)


def test_the_chosen_depth_reaches_the_main_llm():
    jev = _jev(
        {
            "meaning": _a("answerable", 0.9),
            "action_answerable": _a("reply_full", 0.9),
            "effort": _a("high", 0.8),
        }
    )
    d, _ = _run(jev)
    assert (d.branch, d.effort) == ("full", "high")
