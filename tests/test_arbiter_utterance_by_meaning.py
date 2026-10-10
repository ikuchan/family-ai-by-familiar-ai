"""発話は意味 → 動作を先読みの 1 回で聞く（出-ay 段 4-4b・2026-10-09・`設計方針_判定の段` §2.2.5）。

人の言葉への最初の反復は、`core/utterance_meaning` の問い（意味＋意味ごとの動作）を Jev に 1 回で聞き、決まりで最終の動作を
決めて `Decision` に写す。越えなければ「よく考えるか、軽く聞き返すか」をもう 1 回聞く。黙ると決まったら、発話の最初の反復でも
主LLM を呼ばずに閉じる。黙る依頼・解く・名乗り・否定は意味と動作で、時期は recall の語と一緒に軽量LLM が書く（段 4-4c）。
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
        {
            "text": "もう一度言ってもらえますか？",
            "tool_input": {"name": "ケイマン"},
            "query": "明日の天気",
        }
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
    assert (d.branch, d.text) == ("light", "もう一度言ってもらえますか？")
    assert "聞き返す" in prompt_of(writer)


def test_when_unsure_it_asks_think_or_ask_back():
    jev = _jev({"meaning": _a("research", 0.5)}, {"action": _a("reply_full", 0.7)})
    d, _ = _run(jev)
    assert jev.ask.await_count == 2
    assert set(jev.ask.await_args.args[1]["action"]["criteria"]) == {"reply_full", "ask_back"}
    assert d.branch == "full"
    jev = _jev({"meaning": _a("research", 0.5)}, {"action": _a("reply_full", 0.4)})
    d, _ = _run(jev)
    assert (d.branch, d.text) == (
        "light",
        "もう一度言ってもらえますか？",
    )  # それも越えなければ軽く聞き返す


def test_answering_from_what_is_known():
    jev = _jev({"meaning": _a("answerable", 0.9), "action_answerable": _a("reply_light", 0.8)})
    d, _ = _run(jev)
    assert (d.branch, d.text) == ("light", "もう一度言ってもらえますか？")


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


# ── 段 4-4c：黙る依頼・解く・名乗り・否定・時期を、新しい問いで欄に戻す（2026-10-09）──────────────────────


def test_a_quiet_request_sets_the_minutes():
    jev = _jev(
        {
            "meaning": _a("time", 0.9),
            "action_time": _a("quiet", 0.9),
            "quiet_minutes": _a("30", 0.8),
        }
    )
    d, _ = _run(jev)
    assert (d.branch, d.silence_minutes) == ("light", 30)
    assert "quiet_minutes" in jev.ask.await_args.args[1]


def test_a_release_lifts_the_silence():
    jev = _jev({"meaning": _a("time", 0.9), "action_time": _a("lift_quiet", 0.9)})
    d, _ = _run(jev)
    assert d.lift_silence and d.branch == "light"


def test_a_claim_names_the_speaker():
    d, _ = _run(_jev({"meaning": _a("claim", 0.9), "action_claim": _a("パパ", 0.9)}))
    assert (d.speaker_claim, d.branch) == ("パパ", "light")
    d, _ = _run(_jev({"meaning": _a("claim", 0.9), "action_claim": _a("other", 0.9)}))
    assert d.speaker_claim == ""


def test_a_denial_names_who_it_is_not():
    d, _ = _run(_jev({"meaning": _a("deny", 0.9), "action_deny": _a("パパ", 0.9)}))
    assert d.not_person == "パパ"


def test_a_recall_also_writes_the_time_it_points_to():
    jev = _jev({"meaning": _a("research", 0.9), "action_research": _a("recall", 0.9)})
    writer = writer_says(
        {"query": "夏の旅行", "time_ref": "2025-08-15T00:00:00", "time_span_days": 30}
    )
    d, writer = _run(jev, writer=writer)
    assert (d.action, d.time_ref, d.time_span_days) == ("recall", "2025-08-15T00:00:00", 30.0)
    assert "指していなければ空" in prompt_of(writer)


# ── 段 4-4d：聞き返すは 20 字までの指示を付けて軽量LLM に書かせる（2026-10-09）─────────────────────────────


def test_an_ask_back_is_told_to_stay_within_twenty_characters():
    """字数は `utterance_meaning.ASK_BACK_MAX_CHARS`。書けた文が超えても切らずに話す（本人）。"""
    from familiar_agent.core import utterance_meaning as um

    limit = f"{um.ASK_BACK_MAX_CHARS} 字まで"
    jev = _jev({"meaning": _a("music", 0.9), "action_music": _a("ask_back", 0.2)})
    _, writer = _run(jev)
    assert limit in prompt_of(writer)
    jev = _jev({"meaning": _a("research", 0.5)}, {"action": _a("reply_full", 0.4)})
    _, writer = _run(jev)
    assert limit in prompt_of(writer)  # 越えなかったときの軽く聞き返すも同じ


def test_a_long_ask_back_is_spoken_whole():
    jev = _jev({"meaning": _a("music", 0.9), "action_music": _a("ask_back", 0.9)})
    long = "どの曲をかけたらいいか、もう一度だけ教えてもらえますか？"
    d, _ = _run(jev, writer=writer_says({"text": long}))
    assert d.text == long
