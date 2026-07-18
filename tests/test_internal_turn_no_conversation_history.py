"""Tests for internal desire turn conversation history isolation.

非社会的な内的desireターンで stream_turn が受け取る messages に事前の会話履歴が
含まれないことを検証する。

Red/Green 対応:
  test_internal_turn_excludes_conversation_history:
    修正前(3628: list(self.messages)) → FAIL (Red) ★主検証
    修正後([self.messages[-1]])     → PASS (Green)
  test_internal_turn_messages_not_empty:
    修正前後どちらも満たす（"." を含む空contents回避の回帰防止）
  test_normal_turn_keeps_conversation_history:
    修正前後どちらも満たす（通常ターンを壊していないことの確認）
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from familiar_agent.backend import TurnResult
from familiar_agent.exploration import ExplorationTracker
from familiar_agent.mood_register import MoodPAD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _end_turn_result() -> TurnResult:
    return TurnResult(
        stop_reason="end_turn",
        text="done",
        tool_calls=[],
        input_tokens=10,
        output_tokens=5,
    )


async def _fake_end_turn_stream(*, system, messages, tools, max_tokens, on_text=None):
    return _end_turn_result(), None


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def _make_agent(*, separate_utility: bool):
    """Minimal EmbodiedAgent with heavy deps mocked.

    separate_utility=True  → _utility_backend ≠ backend
                             → _maybe_swap_internal_backend が swap し、内的ターン分岐
    separate_utility=False → _utility_backend == backend
                             → swap なし (通常ターン相当)
    """
    from familiar_agent.agent import EmbodiedAgent
    from familiar_agent.self_narrative import SelfNarrative
    from familiar_agent.relationship import RelationshipTracker, PersonRegistry
    from familiar_agent.workspace import GlobalWorkspace
    from familiar_agent.prediction import PredictionEngine
    from familiar_agent.attention_schema import AttentionSchema

    agent = EmbodiedAgent.__new__(EmbodiedAgent)

    agent.config = MagicMock()
    agent.config.max_tokens = 512
    agent.config.agent_name = "Kokone"
    agent.config.companion_name = "Kouta"
    agent.config.auto_say = False
    agent.config.thinking_mode = False

    agent._turn_count = 0
    agent._session_input_tokens = 0
    agent._session_output_tokens = 0
    agent._last_context_tokens = 0
    agent._post_compact = False
    agent._background_tasks = set()
    agent._cached_plan_ctx = ""
    agent._cached_workspace_ctx = ""
    agent._cached_temporal_ctx = None
    agent._cached_companion_mood = "engaged"
    agent._started_at = 0.0
    agent.messages = []
    agent._me_md = ""
    agent._family_md = ""
    agent._presence_watcher = None
    agent._tool_failure_streak = 0
    agent._last_human_at = 0.0
    agent._coherence_retried = False

    def _make_backend(name: str) -> MagicMock:
        b = MagicMock()
        b.complete = AsyncMock(return_value=f"summary-{name}")
        b.make_user_message = lambda t: {"role": "user", "content": t}
        b.make_assistant_message = lambda result, raw: {"role": "assistant", "content": result.text}
        b.make_tool_results = MagicMock(return_value=[{"role": "tool", "content": "ok"}])
        b.stream_turn = AsyncMock(side_effect=_fake_end_turn_stream)
        return b

    main_backend = _make_backend("main")
    agent.backend = main_backend
    if separate_utility:
        agent._utility_backend = _make_backend("utility")
    else:
        agent._utility_backend = main_backend

    mem = MagicMock()
    mem.is_embedding_ready = MagicMock(return_value=True)
    mem.recall_async = AsyncMock(return_value=[])
    mem.recent_feelings_async = AsyncMock(return_value=[])
    mem.recall_self_model_async = AsyncMock(return_value=[])
    mem.recall_curiosities_async = AsyncMock(return_value=[])
    mem.recall_day_summaries_async = AsyncMock(return_value=[])
    mem.recall_semantic_facts_async = AsyncMock(return_value=[])
    mem.recall_behavior_policies_async = AsyncMock(return_value=[])
    mem.format_for_context = MagicMock(return_value="")
    mem.format_feelings_for_context = MagicMock(return_value="")
    mem.format_day_summaries_for_context = MagicMock(return_value="")
    mem.format_semantic_facts_for_context = MagicMock(return_value="")
    mem.format_behavior_policies_for_context = MagicMock(return_value="")
    mem.format_self_model_for_context = MagicMock(return_value="")
    mem.format_curiosities_for_context = MagicMock(return_value="")
    mem.save_async = AsyncMock()
    mem.adjust_semantic_fact_confidence_async = AsyncMock(return_value=None)
    mem.adjust_behavior_policy_confidence_async = AsyncMock(return_value=None)
    mem.get_dates_with_observations = MagicMock(return_value=[])
    mem.get_dates_with_summaries = MagicMock(return_value=[])
    mem.as_coalition_async = AsyncMock(return_value=None)
    mem.pick_seed_candidates = MagicMock(return_value=[])
    agent._memory = mem

    mem_tool = MagicMock()
    mem_tool.get_tool_definitions = MagicMock(return_value=[])
    mem_tool.call = AsyncMock(return_value=("remembered", None))
    mem_tool._notes_registered_this_turn = 0
    mem_tool._pending_store = MagicMock()
    mem_tool._pending_store.list_active = MagicMock(return_value=[])
    agent._memory_tool = mem_tool
    agent._pending_store = mem_tool._pending_store

    tom = MagicMock()
    tom.get_tool_definitions = MagicMock(return_value=[])
    agent._tom_tool = tom

    coding = MagicMock()
    coding.get_tool_definitions = MagicMock(return_value=[])
    agent._coding = coding

    agent._tts = None
    agent._stt = None
    agent._camera = None
    agent._mobility = None
    agent._mcp = None

    def _make_deferred() -> MagicMock:
        d = MagicMock()
        d.get_tool_definitions = MagicMock(return_value=[])
        d.call = AsyncMock(return_value=("", None))
        d.pending_context = MagicMock(return_value="")
        d.has_pending = False
        d.is_running = False
        d.has_user_initiated_pending = False
        d.set_user_turn = MagicMock()
        return d

    agent._deferred_search = _make_deferred()
    agent._deferred_fetch = _make_deferred()

    pmm = MagicMock()
    pmm.get_speaker_memory = MagicMock(return_value=None)
    pmm.get_agent_memory = MagicMock(return_value=mem)
    pmm.current_speaker_id = "person-1"
    pmm.find_person_id_by_name = MagicMock(return_value=None)
    pmm.set_speaker = AsyncMock()
    pmm.get_present_ids = MagicMock(return_value=[])
    pmm.get_all_present_memories = MagicMock(return_value=[])
    pmm.get_person_name = MagicMock(return_value="person")
    agent._pmm = pmm

    agent._exploration = ExplorationTracker()
    agent._scene = None
    agent._self_narrative = SelfNarrative()
    agent._relationship = RelationshipTracker()
    agent._persons = PersonRegistry(default_name="Kouta")
    agent._self_state = MagicMock()
    agent._self_state.snapshot = MagicMock(return_value={"unresolved_tension": 0.2})
    agent._workspace = GlobalWorkspace()
    agent._prediction = PredictionEngine()
    agent._attention_schema = AttentionSchema()
    agent._dmn = MagicMock()
    agent._dmn.wander = AsyncMock(return_value=None)
    agent._meta_monitor = MagicMock()
    agent._meta_monitor.as_coalition = MagicMock(return_value=None)
    agent._meta_monitor.record_step = MagicMock()
    agent._memory_worker = MagicMock()
    agent._memory_worker.is_running = True
    agent._mood = "neutral"
    agent._mood_intensity = 0.0
    agent._mood_set_at = time.time()

    return agent


def _make_patches() -> list:
    return [
        patch("familiar_agent.agent.EmbodiedAgent._morning_reconstruction",
              new=AsyncMock(return_value="")),
        patch("familiar_agent.agent.EmbodiedAgent._infer_companion_mood",
              new=AsyncMock(return_value="engaged")),
        patch("familiar_agent.agent.EmbodiedAgent._emotion_for_turn",
              new=AsyncMock(return_value=(MoodPAD(), "neutral"))),
        patch("familiar_agent.agent.EmbodiedAgent._summarize_exchange",
              new=AsyncMock(return_value="summary")),
        patch("familiar_agent.agent.EmbodiedAgent._online_temporal_context",
              new=AsyncMock(return_value=None)),
        patch("familiar_agent.agent.EmbodiedAgent._run_post_response_pipeline",
              new=AsyncMock()),
        patch("familiar_agent.agent.EmbodiedAgent._update_self_model",
              new=AsyncMock()),
        patch("familiar_agent.agent.EmbodiedAgent._maybe_update_self_narrative",
              new=AsyncMock()),
        patch("familiar_agent.agent.EmbodiedAgent._maybe_adapt_values",
              new=AsyncMock()),
        patch("familiar_agent.agent.EmbodiedAgent.extract_curiosity",
              new=AsyncMock(return_value=None)),
        patch("familiar_agent.agent.EmbodiedAgent._proactive_memory_context",
              new=AsyncMock(return_value=None)),
        patch("familiar_agent.agent.generate_plan",
              new=AsyncMock(return_value="")),
        patch("familiar_agent.agent.check_plan_blocked",
              new=AsyncMock(return_value=False)),
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def internal_agent():
    """utility ≠ main backend の内的ターン用エージェント。"""
    agent = _make_agent(separate_utility=True)
    patches = _make_patches()
    for p in patches:
        p.start()
    yield agent
    for p in patches:
        p.stop()


@pytest.fixture()
def normal_agent():
    """utility == main backend の通常ターン用エージェント。"""
    agent = _make_agent(separate_utility=False)
    patches = _make_patches()
    for p in patches:
        p.start()
    yield agent
    for p in patches:
        p.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_internal_turn_excludes_conversation_history(internal_agent):
    """【主検証・Red】非社会的な内的desireターンで stream_turn が受け取る messages に
    事前の会話履歴が含まれないこと。

    修正前(3628: list(self.messages)) → PRIOR_CONVERSATION_MARKER が含まれ FAIL (Red)
    修正後([self.messages[-1]])       → "." のみで PASS (Green)
    """
    agent = internal_agent
    agent.messages.append(agent.backend.make_user_message("PRIOR_CONVERSATION_MARKER"))

    seen: dict = {}

    async def capturing_stream_turn(*, system, messages, tools, max_tokens, on_text=None):
        seen["messages"] = list(messages)
        return _end_turn_result(), None

    agent._utility_backend.stream_turn = capturing_stream_turn

    await agent.run(user_input="", inner_voice="内省中...", desire_name="look_around")

    flat = repr(seen.get("messages", []))
    assert "PRIOR_CONVERSATION_MARKER" not in flat, (
        "内的ターンが会話履歴を引き継いでいる（直前発話の復唱の原因）"
    )


@pytest.mark.asyncio
async def test_internal_turn_messages_not_empty(internal_agent):
    """内的desireターンで stream_turn が受け取る messages が空でないこと。

    "." プレースホルダが残り、Gemini の空contents拒否を回避。
    修正前後どちらも満たす（回帰防止）。
    """
    agent = internal_agent

    seen: dict = {}

    async def capturing_stream_turn(*, system, messages, tools, max_tokens, on_text=None):
        seen["messages"] = list(messages)
        return _end_turn_result(), None

    agent._utility_backend.stream_turn = capturing_stream_turn

    await agent.run(user_input="", inner_voice="内省中...", desire_name="look_around")

    assert len(seen.get("messages", [])) >= 1, (
        "messages が空（Gemini が空contentsとして拒否しうる）"
    )


@pytest.mark.asyncio
async def test_normal_turn_keeps_conversation_history(normal_agent):
    """通常ターンは会話履歴を従来通り保持すること（変更なしの回帰防止）。

    修正前後どちらも満たす。通常ターンを壊していないことの確認。
    """
    agent = normal_agent
    agent.messages.append(agent.backend.make_user_message("PRIOR_CONVERSATION_MARKER"))

    seen: dict = {}

    async def capturing_stream_turn(*, system, messages, tools, max_tokens, on_text=None):
        seen["messages"] = list(messages)
        return _end_turn_result(), None

    agent.backend.stream_turn = capturing_stream_turn

    await agent.run(user_input="こんにちは")

    assert "PRIOR_CONVERSATION_MARKER" in repr(seen.get("messages", [])), (
        "通常ターンで会話履歴が stream_turn に渡されていない"
    )
