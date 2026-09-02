"""Tests for the EmbodiedAgent ReAct loop (run() method)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from familiar_agent.backend import TurnResult
from familiar_agent.exploration import ExplorationTracker
from familiar_agent.io.aif import AIF
from familiar_agent.mood_register import MoodPAD


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _turn(stop: str, text: str = "", tool_calls: list | None = None) -> TurnResult:
    return TurnResult(
        stop_reason=stop,
        text=text,
        tool_calls=tool_calls or [],
        input_tokens=100,
        output_tokens=50,
    )


def _make_agent(*, with_tts: bool = False, with_camera: bool = False, with_mcp: bool = False):
    """Minimal EmbodiedAgent with all heavy dependencies mocked out."""
    from familiar_agent.agent import EmbodiedAgent

    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    agent.config = MagicMock()
    agent.config.max_tokens = 1000
    agent.config.agent_name = "Kokone"
    agent.config.companion_name = "Kouta"

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

    # Backend: make_tool_results must accept (tool_calls, results) and return a list
    backend = MagicMock()
    backend.complete = AsyncMock(return_value="")
    backend.make_user_message = lambda t: {"role": "user", "content": t}
    backend.make_assistant_message = lambda result, raw: {
        "role": "assistant",
        "content": result.text,
    }
    backend.make_tool_results = MagicMock(return_value=[{"role": "tool", "content": "ok"}])
    agent.backend = backend
    agent._utility_backend = backend

    # Memory
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
    mem.save_async_with_id = AsyncMock(return_value=(None, True))
    mem.content_novelty_async = AsyncMock(return_value=0.5)
    mem.adjust_semantic_fact_confidence_async = AsyncMock(return_value=None)
    mem.adjust_behavior_policy_confidence_async = AsyncMock(return_value=None)
    mem.get_dates_with_observations = MagicMock(return_value=[])
    mem.get_dates_with_summaries = MagicMock(return_value=[])
    mem.as_coalition_async = AsyncMock(return_value=None)
    agent._memory = mem

    mem_tool = MagicMock()
    mem_tool.get_tool_definitions = MagicMock(return_value=[])
    mem_tool.call = AsyncMock(return_value=("remembered", None))
    agent._memory_tool = mem_tool


    coding = MagicMock()
    coding.get_tool_definitions = MagicMock(return_value=[])
    coding.call = AsyncMock(return_value=("code result", None))
    agent._coding = coding

    agent._camera = None
    agent._mobility = None
    agent._mcp = None

    if with_tts:
        tts_tool = MagicMock()
        tts_tool.get_tool_definitions = MagicMock(return_value=[{"name": "say"}])
        tts_tool.call = AsyncMock(return_value=("spoken", None))
        agent._tts = tts_tool
    else:
        agent._tts = None

    if with_camera:
        cam = MagicMock()
        cam.get_tool_definitions = MagicMock(return_value=[{"name": "see"}, {"name": "look"}])
        cam.call = AsyncMock(return_value=("I see a room", "base64img"))
        agent._camera = cam

    if with_mcp:
        mcp_client = MagicMock()
        mcp_client.get_tool_definitions = MagicMock(return_value=[])
        mcp_client.call = AsyncMock(return_value=("mcp result", None))
        mcp_client.is_started = True
        agent._mcp = mcp_client

    deferred = MagicMock()
    deferred.get_tool_definitions = MagicMock(return_value=[])
    deferred.call = AsyncMock(return_value=("search started", None))
    deferred.pending_context = MagicMock(return_value="")
    deferred.has_pending = False
    deferred.is_running = False
    deferred.has_user_initiated_pending = False
    deferred.set_user_turn = MagicMock()
    agent._deferred_search = deferred

    deferred_fetch = MagicMock()
    deferred_fetch.get_tool_definitions = MagicMock(return_value=[])
    deferred_fetch.call = AsyncMock(return_value=("fetch started", None))
    deferred_fetch.pending_context = MagicMock(return_value="")
    deferred_fetch.has_pending = False
    deferred_fetch.is_running = False
    deferred_fetch.has_user_initiated_pending = False
    deferred_fetch.set_user_turn = MagicMock()
    agent._deferred_fetch = deferred_fetch

    mock_pmm = MagicMock()
    mock_pmm.get_speaker_memory = MagicMock(return_value=None)
    mock_pmm.get_agent_memory = MagicMock(return_value=mem)
    mock_pmm.current_speaker_id = None
    mock_pmm.get_present_ids = MagicMock(return_value=[])
    mock_pmm.find_person_id_by_name = MagicMock(return_value=None)
    mock_pmm.set_speaker = AsyncMock()
    agent._pmm = mock_pmm

    agent._exploration = ExplorationTracker()
    agent._scene = None

    from familiar_agent.self_narrative import SelfNarrative
    from familiar_agent.relationship import RelationshipTracker
    from familiar_agent.prediction import PredictionEngine
    import time as _time

    agent._self_narrative = SelfNarrative()
    agent._relationship = RelationshipTracker()
    agent._self_state = MagicMock()
    agent._self_state.snapshot = MagicMock(return_value={"unresolved_tension": 0.2})
    agent._prediction = PredictionEngine()
    # T との行き来の口（`設計図` ③-2）。実物は `__init__` が持たせるが、この
    # ヘルパーは `__new__` で組み立てるので、ここでも持たせる。
    agent._aif = AIF(None)
    agent._memory_worker = MagicMock()
    agent._memory_worker.is_running = True
    agent._mood = "neutral"
    agent._mood_intensity = 0.0
    agent._mood_set_at = _time.time()

    return agent


# Patches that suppress heavy async sub-calls in run()
_HEAVY_PATCHES = {
    "familiar_agent.agent.EmbodiedAgent._infer_companion_mood": AsyncMock(return_value="engaged"),
    "familiar_agent.agent.EmbodiedAgent._emotion_for_turn": AsyncMock(return_value=(MoodPAD(), 0.5, "neutral")),
    "familiar_agent.agent.EmbodiedAgent._summarize_exchange": AsyncMock(return_value="summary"),
    "familiar_agent.agent.EmbodiedAgent._run_post_response_pipeline": AsyncMock(),
    "familiar_agent.agent.EmbodiedAgent._maybe_update_self_narrative": AsyncMock(),
    "familiar_agent.agent.EmbodiedAgent._maybe_adapt_values": AsyncMock(),
    "familiar_agent.agent.EmbodiedAgent.extract_curiosity": AsyncMock(return_value=None),
    "familiar_agent.agent.generate_plan": AsyncMock(return_value=""),
    "familiar_agent.agent.check_plan_blocked": AsyncMock(return_value=False),
}


def _patch_heavy(extra: dict | None = None):
    """Apply all heavy patches; returns a list of patch objects (must be started/stopped by caller)."""
    patches = dict(_HEAVY_PATCHES)
    if extra:
        patches.update(extra)
    return [patch(target, new) for target, new in patches.items()]


# ---------------------------------------------------------------------------
# Tests: basic single-turn end_turn
# ---------------------------------------------------------------------------
















# ---------------------------------------------------------------------------
# Tests: tool_use → end_turn sequence
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# Tests: 発話は say() だけが担う（auto-say 撤去）
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Tests: one-time say() nudge at end_turn (Step 3 of silence-control)
# ---------------------------------------------------------------------------


def _nudge_messages(agent) -> list:
    """Return user messages that look like the say() nudge."""
    out = []
    for m in agent.messages:
        content = m.get("content") if isinstance(m, dict) else None
        if (
            isinstance(content, str)
            and m.get("role") == "user"
            and "say()" in content
            and "声に出して" in content
        ):
            out.append(m)
    return out










# ---------------------------------------------------------------------------
# Tests: morning reconstruction on first turn
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Tests: online temporal self + adaptive values
# ---------------------------------------------------------------------------




@pytest.mark.asyncio
async def test_maybe_update_self_narrative_uses_agency_error_trigger():
    agent = _make_agent()
    agent._utility_backend.complete = AsyncMock(return_value="ウチは少し揺れながら確かめ直した。")
    agent._self_narrative.write = MagicMock()
    agent._prediction.last_signal = MagicMock(
        return_value=SimpleNamespace(action_name="look", agency_error=0.72)
    )

    await agent._maybe_update_self_narrative(
        user_input="何が見えた？",
        final_text="まだ少しずれてる気がする",
        emotion="neutral",
        is_desire_turn=False,
    )

    agent._self_narrative.write.assert_called_once()
    assert agent._self_narrative.write.call_args.kwargs["trigger"] == "agency_error"


@pytest.mark.asyncio
async def test_maybe_adapt_values_updates_curiosity_and_support_policies():
    agent = _make_agent()
    agent._prediction.last_signal = MagicMock(
        return_value=SimpleNamespace(action_name="look", agency_error=0.68)
    )
    desires = MagicMock()
    desires.boost = MagicMock()

    await agent._maybe_adapt_values(
        user_input="大丈夫？",
        final_text="窓の向こうの空が気になったよ。",
        emotion="tender",
        camera_used=True,
        curiosity="窓の向こうの空",
        is_desire_turn=False,
        desires=desires,
    )

    calls = agent._memory.adjust_behavior_policy_confidence_async.await_args_list
    assert len(calls) >= 2
    assert any(call.args[:2] == ("curiosity:active", 0.08) for call in calls)
    assert any(call.args[:2] == ("curiosity:active", -0.05) for call in calls)
    assert any(call.args[:2] == ("conversation:supportive_style", 0.04) for call in calls)
    desires.boost.assert_called_once_with("share_memory", 0.08)


# ---------------------------------------------------------------------------
# Tests: empty / edge cases
# ---------------------------------------------------------------------------








@pytest.mark.asyncio
async def test_post_response_pipeline_updates_concerns():
    from familiar_agent.agent import EmbodiedAgent

    agent = _make_agent()
    agent._concerns = MagicMock()
    agent._prediction.last_signal = MagicMock(
        return_value=SimpleNamespace(
            action_name="look",
            agency_error=0.62,
            external_surprise=0.18,
        )
    )
    agent._emotion_for_turn = AsyncMock(return_value=(MoodPAD(), 0.5, "tender"))
    agent._summarize_exchange = AsyncMock(return_value="summary")
    agent._maybe_update_self_narrative = AsyncMock()
    agent._maybe_adapt_values = AsyncMock()
    agent.extract_curiosity = AsyncMock(return_value="The window light still feels important.")

    desires = MagicMock()
    desires.boost = MagicMock()
    desires.curiosity_target = None

    await EmbodiedAgent._run_post_response_pipeline(
        agent,
        user_input="どう見えた？",
        final_text="窓の光が少し気になってる。",
        camera_used=True,
        camera_image=None,
        observation_action_name="look",
        observation_action_input={"direction": "left", "degrees": 30},
        companion_mood="frustrated",
        is_desire_turn=False,
        desires=desires,
    )

    agent._concerns.update_from_turn.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: deferred search result delivery inner_voice injection
# ---------------------------------------------------------------------------


def _system_text(system: str | tuple) -> str:
    """Flatten the (stable, variable) tuple returned by _system_prompt() into one string."""
    if isinstance(system, tuple):
        return "\n\n---\n\n".join(s for s in system if s)
    return system






# ---------------------------------------------------------------------------
# Quiet-hours bypass for share_search_result (user-initiated)
# ---------------------------------------------------------------------------


def _make_quiet_rule():
    rule = MagicMock()
    rule.is_quiet = MagicMock(return_value=True)
    rule.start_hour = 21
    rule.end_hour = 5
    return rule






# ---------------------------------------------------------------------------
# 静穏時間の判定に使う共通ヘルパー
# ---------------------------------------------------------------------------


def _make_active_rule():
    """A schedule rule that is never in quiet hours."""
    rule = MagicMock()
    rule.is_quiet = MagicMock(return_value=False)
    rule.start_hour = 23
    rule.end_hour = 7
    return rule


class TestInQuietHoursHelper:
    def test_true_when_schedule_rule_is_quiet(self):
        from familiar_agent.agent import EmbodiedAgent

        agent = EmbodiedAgent.__new__(EmbodiedAgent)
        agent._schedule_rule = _make_quiet_rule()
        assert agent._in_quiet_hours() is True

    def test_false_when_schedule_rule_not_quiet(self):
        from familiar_agent.agent import EmbodiedAgent

        agent = EmbodiedAgent.__new__(EmbodiedAgent)
        agent._schedule_rule = _make_active_rule()
        assert agent._in_quiet_hours() is False

    def test_false_when_no_schedule_rule(self):
        from familiar_agent.agent import EmbodiedAgent

        agent = EmbodiedAgent.__new__(EmbodiedAgent)
        assert agent._in_quiet_hours() is False





# ---------------------------------------------------------------------------
# Internal desire turns pass Anthropic-format messages to utility backend
# (format conversion is handled inside each backend's stream_turn, not at agent level)
# ---------------------------------------------------------------------------






@pytest.mark.asyncio
async def test_pipeline_supersedes_loop_obs_without_camera():
    """イベントループのターン（camera_used=False）でもループ中 O の後始末が走る。

    supersede をカメラ分岐の中に置いていたため、イベントループのターンでは一度も走らず、
    トリガ O が残り続けて W を汚した（実機で観測）。カメラ無しでも会話 O で始末する。
    """
    from familiar_agent.agent import EmbodiedAgent

    agent = _make_agent()
    agent._emotion_for_turn = AsyncMock(return_value=(MoodPAD(), 0.5, "tender"))
    agent._summarize_exchange = AsyncMock(return_value="summary")
    agent._maybe_update_self_narrative = AsyncMock()
    agent._maybe_adapt_values = AsyncMock()
    agent._active_memory = MagicMock(return_value=agent._memory)
    agent._memory.save_async_with_id = AsyncMock(return_value=("conv-1", True))
    agent._memory.mark_superseded = MagicMock()

    await EmbodiedAgent._run_post_response_pipeline(
        agent,
        user_input="昨日の天気覚えてる？",
        final_text="晴れてたよ",
        camera_used=False,
        camera_image=None,
        observation_action_name=None,
        observation_action_input=None,
        companion_mood="engaged",
        is_desire_turn=False,
        desires=None,
        superseded_ids=["loop-1", "loop-2"],
    )

    calls = [c.args for c in agent._memory.mark_superseded.call_args_list]
    assert ("loop-1", "conv-1") in calls
    assert ("loop-2", "conv-1") in calls
