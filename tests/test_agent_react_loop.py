"""Tests for the EmbodiedAgent ReAct loop (run() method)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from familiar_agent.backend import ToolCall, TurnResult
from familiar_agent.desires import DesireSystem
from familiar_agent.exploration import ExplorationTracker


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

    tom = MagicMock()
    tom.get_tool_definitions = MagicMock(return_value=[])
    tom.call = AsyncMock(return_value=("tom result", None))
    agent._tom_tool = tom

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
    mock_pmm.find_person_id_by_name = MagicMock(return_value=None)
    mock_pmm.set_speaker = AsyncMock()
    agent._pmm = mock_pmm

    agent._exploration = ExplorationTracker()
    agent._scene = None

    from familiar_agent.self_narrative import SelfNarrative
    from familiar_agent.relationship import RelationshipTracker
    from familiar_agent.workspace import GlobalWorkspace
    from familiar_agent.prediction import PredictionEngine
    from familiar_agent.attention_schema import AttentionSchema
    import time as _time

    agent._self_narrative = SelfNarrative()
    agent._relationship = RelationshipTracker()
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
    agent._mood_set_at = _time.time()

    return agent


# Patches that suppress heavy async sub-calls in run()
_HEAVY_PATCHES = {
    "familiar_agent.agent.EmbodiedAgent._morning_reconstruction": AsyncMock(return_value=""),
    "familiar_agent.agent.EmbodiedAgent._infer_companion_mood": AsyncMock(return_value="engaged"),
    "familiar_agent.agent.EmbodiedAgent._infer_emotion": AsyncMock(return_value="neutral"),
    "familiar_agent.agent.EmbodiedAgent._summarize_exchange": AsyncMock(return_value="summary"),
    "familiar_agent.agent.EmbodiedAgent._online_temporal_context": AsyncMock(return_value=None),
    "familiar_agent.agent.EmbodiedAgent._run_post_response_pipeline": AsyncMock(),
    "familiar_agent.agent.EmbodiedAgent._update_self_model": AsyncMock(),
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


@pytest.mark.asyncio
async def test_run_end_turn_returns_text():
    """run() with immediate end_turn returns the model's text."""
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text="Hello!"), "Hello!"))

    ps = _patch_heavy()
    for p in ps:
        p.start()
    try:
        result = await agent.run("こんにちは")
    finally:
        for p in ps:
            p.stop()

    assert result == "Hello!"


@pytest.mark.asyncio
async def test_brief_greeting_turn_uses_only_say_and_skips_heavy_prep():
    agent = _make_agent(with_tts=True, with_camera=True)
    agent.backend.stream_turn = AsyncMock(
        return_value=(_turn("end_turn", text="おはよう。"), "おはよう。")
    )

    morning_mock = AsyncMock(return_value="morning context")
    companion_mood_mock = AsyncMock(return_value="engaged")
    workspace_mock = AsyncMock(return_value="[workspace]")
    patches = dict(_HEAVY_PATCHES)
    patches["familiar_agent.agent.EmbodiedAgent._morning_reconstruction"] = morning_mock
    patches["familiar_agent.agent.EmbodiedAgent._infer_companion_mood"] = companion_mood_mock
    patches["familiar_agent.agent.EmbodiedAgent._gather_workspace_context"] = workspace_mock

    ps = [patch(t, n) for t, n in patches.items()]
    for p in ps:
        p.start()
    try:
        result = await agent.run("おはよう")
    finally:
        for p in ps:
            p.stop()

    assert result == "おはよう。"
    stream_kwargs = agent.backend.stream_turn.await_args.kwargs
    assert stream_kwargs["tools"] == [{"name": "say"}]
    assert stream_kwargs["max_tokens"] == 120
    morning_mock.assert_not_awaited()
    companion_mood_mock.assert_not_awaited()
    workspace_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_increments_turn_count():
    """run() increments _turn_count on each invocation."""
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text="Hi"), "Hi"))

    ps = _patch_heavy()
    for p in ps:
        p.start()
    try:
        assert agent._turn_count == 0
        await agent.run("test")
        assert agent._turn_count == 1
        await agent.run("test2")
        assert agent._turn_count == 2
    finally:
        for p in ps:
            p.stop()


@pytest.mark.asyncio
async def test_run_appends_user_message_to_history():
    """run() appends the user message to agent.messages."""
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(
        return_value=(_turn("end_turn", text="response"), "response")
    )

    ps = _patch_heavy()
    for p in ps:
        p.start()
    try:
        assert len(agent.messages) == 0
        await agent.run("hello from user")
        # At minimum, a user message was added
        assert any(m.get("role") == "user" for m in agent.messages)
    finally:
        for p in ps:
            p.stop()


@pytest.mark.asyncio
async def test_repeated_tool_failure_raises_self_protect_without_irritable_tone(tmp_path):
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(
        return_value=(_turn("end_turn", text="落ち着いて進めよう。"), "落ち着いて進めよう。")
    )
    agent._tool_failure_streak = 3
    desires = DesireSystem(state_path=tmp_path / "desires.json", companion_name="Kota")

    ps = _patch_heavy()
    for p in ps:
        p.start()
    try:
        result = await agent.run("助けて", desires=desires)
    finally:
        for p in ps:
            p.stop()

    assert desires.level("self_protect") > 0.0
    assert "ugh" not in result.lower()
    assert "annoy" not in result.lower()


@pytest.mark.asyncio
async def test_existing_no_hardware_mode_still_works_with_mental_pipeline():
    agent = _make_agent(with_tts=False, with_camera=False, with_mcp=False)
    agent.backend.stream_turn = AsyncMock(
        return_value=(_turn("end_turn", text="hardwareなしでも動く"), "hardwareなしでも動く")
    )

    ps = _patch_heavy()
    for p in ps:
        p.start()
    try:
        result = await agent.run("こんにちは")
    finally:
        for p in ps:
            p.stop()

    assert result == "hardwareなしでも動く"


@pytest.mark.asyncio
async def test_run_accumulates_tokens():
    """run() adds input/output tokens to session totals."""
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(
        return_value=(
            _turn(
                "end_turn",
                text="ok",
            ),
            "ok",
        )
    )
    # Override to set token counts
    result_obj = TurnResult(stop_reason="end_turn", text="ok", input_tokens=200, output_tokens=80)
    agent.backend.stream_turn = AsyncMock(return_value=(result_obj, "ok"))

    ps = _patch_heavy()
    for p in ps:
        p.start()
    try:
        await agent.run("test")
    finally:
        for p in ps:
            p.stop()

    assert agent._session_input_tokens == 200
    assert agent._session_output_tokens == 80


# ---------------------------------------------------------------------------
# Tests: tool_use → end_turn sequence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_tool_use_then_end_turn():
    """run() executes a tool call then gets end_turn on the next iteration."""
    agent = _make_agent()

    tc = ToolCall(id="tc1", name="remember", input={"content": "test memory"})
    turn1 = TurnResult(stop_reason="tool_use", text="", tool_calls=[tc])
    turn2 = TurnResult(stop_reason="end_turn", text="Done!", tool_calls=[])

    agent.backend.stream_turn = AsyncMock(
        side_effect=[
            (turn1, None),
            (turn2, "Done!"),
        ]
    )

    ps = _patch_heavy()
    for p in ps:
        p.start()
    try:
        result = await agent.run("remember something")
    finally:
        for p in ps:
            p.stop()

    assert result == "Done!"
    assert agent._memory_tool.call.called


@pytest.mark.asyncio
async def test_run_tool_results_added_to_messages():
    """Tool results are added to message history after tool execution."""
    agent = _make_agent()

    tc = ToolCall(id="tc1", name="remember", input={"content": "hi"})
    turn1 = TurnResult(stop_reason="tool_use", text="", tool_calls=[tc])
    turn2 = TurnResult(stop_reason="end_turn", text="Saved.", tool_calls=[])

    agent.backend.stream_turn = AsyncMock(
        side_effect=[
            (turn1, None),
            (turn2, "Saved."),
        ]
    )

    ps = _patch_heavy()
    for p in ps:
        p.start()
    try:
        await agent.run("please remember")
    finally:
        for p in ps:
            p.stop()

    # make_tool_results was called with the tool call and its result
    assert agent.backend.make_tool_results.called


@pytest.mark.asyncio
async def test_run_passes_latest_pre_see_action_into_scene_update():
    """The last embodied action before see() conditions the scene update."""
    from familiar_agent.agent import EmbodiedAgent

    agent = _make_agent(with_camera=True)
    agent._scene = MagicMock()
    agent._scene.update = AsyncMock(return_value=[])
    agent._scene.context_for_prompt = MagicMock(return_value="")
    agent._scene_backend = MagicMock()
    agent._camera.call = AsyncMock(
        side_effect=[
            ("looked left", None),
            ("I see a room", "base64img"),
        ]
    )

    turn1 = TurnResult(
        stop_reason="tool_use",
        text="",
        tool_calls=[
            ToolCall(id="tc1", name="look", input={"direction": "left", "degrees": 45}),
            ToolCall(id="tc2", name="see", input={}),
        ],
    )
    turn2 = TurnResult(stop_reason="end_turn", text="There is a window.", tool_calls=[])
    agent.backend.stream_turn = AsyncMock(
        side_effect=[(turn1, None), (turn2, "There is a window.")]
    )

    patches = dict(_HEAVY_PATCHES)
    patches["familiar_agent.agent.EmbodiedAgent._run_post_response_pipeline"] = (
        EmbodiedAgent._run_post_response_pipeline
    )

    ps = [patch(t, n) for t, n in patches.items()]
    for p in ps:
        p.start()
    try:
        await agent.run("look and report")
        await agent._drain_background_tasks(timeout=0.5)
    finally:
        for p in ps:
            p.stop()

    agent._scene.update.assert_awaited_once()
    _, kwargs = agent._scene.update.call_args
    assert kwargs["action_name"] == "look"
    assert kwargs["action_input"] == {"direction": "left", "degrees": 45}


# ---------------------------------------------------------------------------
# Tests: auto-say
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_auto_say_fires_when_tts_available_and_no_say_call():
    """When TTS is present and model wrote text without calling say(), auto-say fires."""
    agent = _make_agent(with_tts=True)
    agent.backend.stream_turn = AsyncMock(
        return_value=(_turn("end_turn", text="Hello, I speak!"), "Hello, I speak!")
    )

    ps = _patch_heavy()
    for p in ps:
        p.start()
    try:
        await agent.run("speak to me")
    finally:
        for p in ps:
            p.stop()

    agent._tts.call.assert_awaited_once()
    call_args = agent._tts.call.call_args
    assert call_args[0][0] == "say"


@pytest.mark.asyncio
async def test_run_no_auto_say_when_tts_absent():
    """Without TTS, no auto-say even if model wrote text."""
    agent = _make_agent(with_tts=False)
    agent.backend.stream_turn = AsyncMock(
        return_value=(_turn("end_turn", text="Silent response"), "Silent response")
    )

    ps = _patch_heavy()
    for p in ps:
        p.start()
    try:
        await agent.run("respond")
    finally:
        for p in ps:
            p.stop()

    assert agent._tts is None


@pytest.mark.asyncio
async def test_run_no_auto_say_when_say_already_called():
    """If say() was called as a tool, auto-say should NOT fire again."""
    agent = _make_agent(with_tts=True)

    tc = ToolCall(id="tc1", name="say", input={"text": "I spoke"})
    turn1 = TurnResult(stop_reason="tool_use", text="", tool_calls=[tc])
    turn2 = TurnResult(stop_reason="end_turn", text="done", tool_calls=[])

    agent.backend.stream_turn = AsyncMock(
        side_effect=[
            (turn1, None),
            (turn2, "done"),
        ]
    )

    ps = _patch_heavy()
    for p in ps:
        p.start()
    try:
        await agent.run("speak via tool")
    finally:
        for p in ps:
            p.stop()

    # say() was called once via tool execution; auto-say must NOT add a second call
    assert agent._tts.call.call_count == 1


# ---------------------------------------------------------------------------
# Tests: morning reconstruction on first turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_first_turn_calls_morning_reconstruction():
    """On the very first turn, _morning_reconstruction is invoked."""
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(
        return_value=(_turn("end_turn", text="Good morning"), "Good morning")
    )

    morning_mock = AsyncMock(return_value="morning context")
    patches = dict(_HEAVY_PATCHES)
    patches["familiar_agent.agent.EmbodiedAgent._morning_reconstruction"] = morning_mock

    ps = [patch(t, n) for t, n in patches.items()]
    for p in ps:
        p.start()
    try:
        await agent.run("今日はどう？")
    finally:
        for p in ps:
            p.stop()

    morning_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_subsequent_turns_skip_morning_reconstruction():
    """_morning_reconstruction is NOT called on turns after the first."""
    agent = _make_agent()
    agent._turn_count = 5  # simulate subsequent turn
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text="reply"), "reply"))

    morning_mock = AsyncMock(return_value="")
    patches = dict(_HEAVY_PATCHES)
    patches["familiar_agent.agent.EmbodiedAgent._morning_reconstruction"] = morning_mock

    ps = [patch(t, n) for t, n in patches.items()]
    for p in ps:
        p.start()
    try:
        await agent.run("follow up")
    finally:
        for p in ps:
            p.stop()

    morning_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# Tests: online temporal self + adaptive values
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_injects_online_temporal_context_into_user_message():
    agent = _make_agent()
    agent._turn_count = 1  # next run is a normal turn, not the first turn
    # Temporal context is now read from cache (populated by post-response pipeline)
    agent._cached_temporal_ctx = "[Temporal self]\n[Resurfaced memory]: 朝の空を探した"
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text="reply"), "reply"))

    ps = [patch(t, n) for t, n in _HEAVY_PATCHES.items()]
    for p in ps:
        p.start()
    try:
        await agent.run("今日はどう？")
    finally:
        for p in ps:
            p.stop()

    user_messages = [m["content"] for m in agent.messages if m.get("role") == "user"]
    assert any("[Temporal self]" in msg for msg in user_messages)


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
async def test_run_empty_text_returns_no_response_placeholder():
    """When model returns no text, run() returns the placeholder string."""
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text=""), ""))

    ps = _patch_heavy()
    for p in ps:
        p.start()
    try:
        result = await agent.run("hi")
    finally:
        for p in ps:
            p.stop()

    assert result == "(no response)"


@pytest.mark.asyncio
async def test_run_schedules_post_response_pipeline_without_blocking_reply():
    """Post-response work should happen in the background after the reply is ready."""
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text="Hello!"), "Hello!"))

    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_pipeline(self, **kwargs):  # noqa: ARG001
        started.set()
        await release.wait()

    patches = dict(_HEAVY_PATCHES)
    patches["familiar_agent.agent.EmbodiedAgent._run_post_response_pipeline"] = _slow_pipeline

    ps = [patch(t, n) for t, n in patches.items()]
    for p in ps:
        p.start()
    try:
        result = await agent.run("こんにちは")
        assert result == "Hello!"
        await asyncio.wait_for(started.wait(), timeout=0.5)
        assert any(not task.done() for task in agent._background_tasks)
    finally:
        release.set()
        await agent._drain_background_tasks(timeout=0.5)
        for p in ps:
            p.stop()


@pytest.mark.asyncio
async def test_run_skips_tape_plan_when_no_separate_utility_backend():
    """No separate utility backend -> skip the extra TAPE planning round-trip."""
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text="Hello!"), "Hello!"))

    plan_mock = AsyncMock(return_value="1. say")
    patches = dict(_HEAVY_PATCHES)
    patches["familiar_agent.agent.generate_plan"] = plan_mock

    ps = [patch(t, n) for t, n in patches.items()]
    for p in ps:
        p.start()
    try:
        await agent.run("こんにちは")
    finally:
        for p in ps:
            p.stop()

    plan_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_response_pipeline_updates_self_continuity_state():
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
    agent._infer_emotion = AsyncMock(return_value="tender")
    agent._summarize_exchange = AsyncMock(return_value="summary")
    agent._update_self_model = AsyncMock()
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
    agent._self_state.apply_turn_context.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: deferred search result delivery inner_voice injection
# ---------------------------------------------------------------------------


def _system_text(system: str | tuple) -> str:
    """Flatten the (stable, variable) tuple returned by _system_prompt() into one string."""
    if isinstance(system, tuple):
        return "\n\n---\n\n".join(s for s in system if s)
    return system


@pytest.mark.asyncio
async def test_run_sets_inner_voice_when_deferred_results_arrive_with_user_message():
    """When deferred search results are available alongside a user message,
    run() should inject a delivery inner_voice so the LLM explicitly reports them."""
    agent = _make_agent()
    agent._deferred_search.pending_context = MagicMock(return_value="【検索結果】日本の最新ニュース…")
    agent.backend.stream_turn = AsyncMock(
        return_value=(_turn("end_turn", text="ニュースが届いたよ！"), "ニュースが届いたよ！")
    )

    captured_system: list = []

    original_stream_turn = agent.backend.stream_turn

    async def _capture_system(*args, system="", **kwargs):
        captured_system.append(system)
        return await original_stream_turn(*args, system=system, **kwargs)

    agent.backend.stream_turn = _capture_system

    ps = _patch_heavy()
    for p in ps:
        p.start()
    try:
        await agent.run("最新ニュース教えて")
    finally:
        for p in ps:
            p.stop()

    assert captured_system, "stream_turn was not called"
    system_text = _system_text(captured_system[0])
    assert "調べておいた結果が届いた" in system_text, (
        "Expected delivery inner_voice in system prompt when deferred results are available"
    )


@pytest.mark.asyncio
async def test_run_does_not_overwrite_explicit_inner_voice_when_deferred_results_arrive():
    """If inner_voice is already provided (e.g. proactive delivery turn), it must not be
    replaced by the auto-generated delivery inner_voice."""
    agent = _make_agent()
    agent._deferred_search.pending_context = MagicMock(return_value="【検索結果】日本の最新ニュース…")
    agent.backend.stream_turn = AsyncMock(
        return_value=(_turn("end_turn", text="届いたよ"), "届いたよ")
    )

    captured_system: list = []

    original_stream_turn = agent.backend.stream_turn

    async def _capture_system(*args, system="", **kwargs):
        captured_system.append(system)
        return await original_stream_turn(*args, system=system, **kwargs)

    agent.backend.stream_turn = _capture_system

    explicit_inner_voice = "「最新ニュース」の検索結果が届いた。いつものトーンで自然に報告しよう。"

    ps = _patch_heavy()
    for p in ps:
        p.start()
    try:
        await agent.run("", inner_voice=explicit_inner_voice)
    finally:
        for p in ps:
            p.stop()

    assert captured_system, "stream_turn was not called"
    system_text = _system_text(captured_system[0])
    assert explicit_inner_voice in system_text, "Explicit inner_voice should be present in system"
    assert "調べておいた結果が届いた" not in system_text, (
        "Auto inner_voice must not overwrite an already-set inner_voice"
    )


# ---------------------------------------------------------------------------
# Quiet-hours bypass for share_search_result (user-initiated)
# ---------------------------------------------------------------------------


def _make_quiet_rule():
    rule = MagicMock()
    rule.is_quiet = MagicMock(return_value=True)
    rule.start_hour = 21
    rule.end_hour = 5
    return rule


@pytest.mark.asyncio
async def test_share_search_result_bypasses_quiet_hours_when_user_initiated():
    """share_search_result delivery must proceed during quiet hours if the search was user-initiated."""
    agent = _make_agent()
    agent._schedule_rule = _make_quiet_rule()
    agent._deferred_search.has_user_initiated_pending = True
    agent.backend.stream_turn = AsyncMock(
        return_value=(_turn("end_turn", text="ニュースが届いたよ！"), "ニュースが届いたよ！")
    )

    ps = _patch_heavy()
    for p in ps:
        p.start()
    try:
        result = await agent.run(
            "",
            inner_voice="調べておいた結果が届いた。いつものトーンで自然に伝えよう。",
            desire_name="share_search_result",
        )
    finally:
        for p in ps:
            p.stop()

    assert result != "", (
        "share_search_result should not be suppressed during quiet hours when user-initiated"
    )
    assert "ニュース" in result


@pytest.mark.asyncio
async def test_share_search_result_suppressed_during_quiet_hours_when_not_user_initiated():
    """share_search_result must still be suppressed during quiet hours for autonomous searches."""
    agent = _make_agent()
    agent._schedule_rule = _make_quiet_rule()
    agent._deferred_search.has_user_initiated_pending = False
    agent._deferred_fetch.has_user_initiated_pending = False

    ps = _patch_heavy()
    for p in ps:
        p.start()
    try:
        result = await agent.run(
            "",
            inner_voice="何か調べた結果が届いた。",
            desire_name="share_search_result",
        )
    finally:
        for p in ps:
            p.stop()

    assert result == "", (
        "Autonomous share_search_result should be suppressed during quiet hours"
    )


# ---------------------------------------------------------------------------
# Internal desire turns pass Gemini-formatted messages to utility backend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_internal_desire_turn_passes_gemini_formatted_messages_to_utility_backend():
    """When swapped to the Gemini utility backend for an internal desire turn,
    messages must be converted to Gemini format (parts, not content) before
    stream_turn is called — no Anthropic-format messages should leak through."""
    import logging
    agent = _make_agent()

    # Give agent a separate utility backend so the swap actually happens
    utility_backend = MagicMock()
    utility_backend.stream_turn = AsyncMock(
        return_value=(_turn("end_turn", text="interesting discovery"), "interesting discovery")
    )
    utility_backend.make_assistant_message = MagicMock(
        return_value={"role": "model", "parts": [{"text": "interesting discovery"}]}
    )
    agent._utility_backend = utility_backend

    # Pre-populate messages in Anthropic format (simulates prior conversation turns)
    agent.messages = [
        {"role": "user", "content": "こんにちは"},
        {"role": "assistant", "content": "こんにちは！"},
    ]

    ps = _patch_heavy()
    for p in ps:
        p.start()
    try:
        await agent.run("", inner_voice="好奇心が高まっている", desire_name="curiosity")
    finally:
        for p in ps:
            p.stop()

    assert utility_backend.stream_turn.called, "utility backend stream_turn should have been called"
    call_kwargs = utility_backend.stream_turn.call_args
    messages_passed = call_kwargs.kwargs.get("messages") or call_kwargs.args[1]

    # Every message must be in Gemini format (has 'parts', not 'content')
    for msg in messages_passed:
        assert "parts" in msg, (
            f"Message passed to utility backend must use Gemini 'parts' format, got: {msg}"
        )
        assert "content" not in msg, (
            f"Anthropic 'content' key must not appear in messages for utility backend, got: {msg}"
        )


@pytest.mark.asyncio
async def test_internal_desire_turn_no_coercion_warning(caplog):
    """No 'coerced' warning should be logged when an internal desire turn runs."""
    import logging
    agent = _make_agent()

    utility_backend = MagicMock()
    utility_backend.stream_turn = AsyncMock(
        return_value=(_turn("end_turn", text="reflecting..."), "reflecting...")
    )
    utility_backend.make_assistant_message = MagicMock(
        return_value={"role": "model", "parts": [{"text": "reflecting..."}]}
    )
    agent._utility_backend = utility_backend

    agent.messages = [
        {"role": "user", "content": "今日はどうだった？"},
        {"role": "assistant", "content": "楽しかったよ。"},
    ]

    ps = _patch_heavy()
    for p in ps:
        p.start()
    try:
        with caplog.at_level(logging.WARNING, logger="familiar_agent.backend"):
            await agent.run("", inner_voice="振り返りたい", desire_name="reflect")
    finally:
        for p in ps:
            p.stop()

    assert "coerced" not in caplog.text, (
        "No coercion warning should be logged for internal desire turns"
    )
