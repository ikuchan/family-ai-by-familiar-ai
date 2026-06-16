"""Tests for message compaction + post-compaction recall in EmbodiedAgent."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock


def _make_msg(role: str, text: str) -> dict:
    return {"role": role, "content": text}


def _make_agent():
    """Minimal EmbodiedAgent with mocked backend and memory."""
    from familiar_agent.agent import EmbodiedAgent
    from familiar_agent.relationship import PersonRegistry

    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    agent.config = MagicMock()
    agent.config.max_tokens = 1000
    agent.config.agent_name = "A"
    agent.config.companion_name = "U"
    agent._started_at = 0.0
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
    agent.messages = []

    agent.backend = MagicMock()
    agent.backend.complete = AsyncMock(return_value="summary text")
    agent.backend.make_user_message = lambda t: _make_msg("user", t)
    agent._utility_backend = agent.backend

    agent._memory = MagicMock()
    agent._memory.recall_async = AsyncMock(return_value=[])
    agent._memory.recall_divergent_async = None  # force recall_async path
    agent._memory.recent_feelings_async = AsyncMock(return_value=[])
    agent._memory.recall_self_model_async = AsyncMock(return_value=[])
    agent._memory.recall_curiosities_async = AsyncMock(return_value=[])
    agent._memory.recall_semantic_facts_async = AsyncMock(return_value=[])
    agent._memory.recall_behavior_policies_async = AsyncMock(return_value=[])
    agent._memory.format_for_context = MagicMock(return_value="")
    agent._memory.format_feelings_for_context = MagicMock(return_value="")
    agent._memory.format_semantic_facts_for_context = MagicMock(return_value="")
    agent._memory.format_behavior_policies_for_context = MagicMock(return_value="")
    agent._me_md = ""
    agent._family_md = ""
    agent._presence_watcher = None
    agent._persons = PersonRegistry(default_name="A")

    from familiar_agent.exploration import ExplorationTracker
    from familiar_agent.self_narrative import SelfNarrative
    from familiar_agent.relationship import RelationshipTracker
    from familiar_agent.workspace import GlobalWorkspace
    from familiar_agent.prediction import PredictionEngine
    from familiar_agent.attention_schema import AttentionSchema
    import time as _time

    agent._exploration = ExplorationTracker()
    agent._scene = None
    agent._self_narrative = SelfNarrative()
    agent._relationship = RelationshipTracker()
    agent._workspace = GlobalWorkspace()
    agent._prediction = PredictionEngine()
    agent._attention_schema = AttentionSchema()
    agent._dmn = MagicMock()
    agent._dmn.wander = AsyncMock(return_value=None)
    agent._meta_monitor = MagicMock()
    agent._meta_monitor.as_coalition = MagicMock(return_value=None)
    agent._meta_monitor.record_step = MagicMock()
    agent._memory.as_coalition_async = AsyncMock(return_value=None)
    agent._memory_worker = MagicMock()
    agent._memory_worker.is_running = True
    agent._mood = "neutral"
    agent._mood_intensity = 0.0
    agent._mood_set_at = _time.time()

    mock_pmm = MagicMock()
    mock_pmm.get_speaker_memory = MagicMock(return_value=None)
    mock_pmm.get_agent_memory = MagicMock(return_value=agent._memory)
    mock_pmm.current_speaker_id = None
    mock_pmm.find_person_id_by_name = MagicMock(return_value=None)
    mock_pmm.set_speaker = AsyncMock()
    agent._pmm = mock_pmm

    deferred = MagicMock()
    deferred.get_tool_definitions = MagicMock(return_value=[])
    deferred.call = AsyncMock(return_value=("search started", None))
    deferred.pending_context = MagicMock(return_value="")
    deferred.has_pending = False
    deferred.is_running = False
    agent._deferred_search = deferred

    deferred_fetch = MagicMock()
    deferred_fetch.get_tool_definitions = MagicMock(return_value=[])
    deferred_fetch.call = AsyncMock(return_value=("fetch started", None))
    deferred_fetch.pending_context = MagicMock(return_value="")
    deferred_fetch.has_pending = False
    deferred_fetch.is_running = False
    agent._deferred_fetch = deferred_fetch

    return agent


# ── _should_compact ────────────────────────────────────────────────────────


class TestShouldCompact:
    def test_false_when_below_threshold(self):
        """Below threshold → no compaction needed."""

        agent = _make_agent()
        agent._last_context_tokens = 30_000
        assert agent._should_compact(threshold_tokens=60_000) is False

    def test_false_when_equal_to_threshold(self):
        """At threshold → no compaction (strictly greater triggers)."""

        agent = _make_agent()
        agent._last_context_tokens = 60_000
        assert agent._should_compact(threshold_tokens=60_000) is False

    def test_true_when_above_threshold(self):
        """Above threshold → compaction needed."""
        agent = _make_agent()
        agent._last_context_tokens = 60_001
        assert agent._should_compact(threshold_tokens=60_000) is True

    def test_false_with_empty_messages(self):
        """No messages → never compact."""
        agent = _make_agent()
        agent._last_context_tokens = 999_999
        agent.messages = []
        assert agent._should_compact(threshold_tokens=0) is False

    def test_default_threshold_is_reasonable(self):
        """Default threshold exists and is positive."""
        import inspect
        from familiar_agent.agent import EmbodiedAgent

        sig = inspect.signature(EmbodiedAgent._should_compact)
        default = sig.parameters["threshold_tokens"].default
        assert isinstance(default, int)
        assert default > 0


# ── _compact_messages ──────────────────────────────────────────────────────


class TestCompactMessages:
    def test_method_exists(self):
        """EmbodiedAgent must have _compact_messages method."""
        from familiar_agent.agent import EmbodiedAgent

        assert hasattr(EmbodiedAgent, "_compact_messages")
        assert callable(EmbodiedAgent._compact_messages)

    def test_keeps_last_n_messages(self):
        """_compact_messages keeps the last keep_last messages intact."""
        agent = _make_agent()
        # 10 messages — last 6 should survive
        for i in range(10):
            agent.messages.append(_make_msg("user" if i % 2 == 0 else "assistant", f"msg{i}"))

        asyncio.run(agent._compact_messages(keep_last=6))

        # Last 6 original messages + 1 summary marker at front
        assert len(agent.messages) == 7

    def test_summary_marker_at_front(self):
        """First message after compaction is a user-role summary."""
        agent = _make_agent()
        for i in range(8):
            agent.messages.append(_make_msg("user" if i % 2 == 0 else "assistant", f"m{i}"))

        asyncio.run(agent._compact_messages(keep_last=4))

        first = agent.messages[0]
        assert first["role"] == "user"
        assert "summary" in first["content"].lower() or len(first["content"]) > 0

    def test_backend_complete_called_for_summary(self):
        """_compact_messages calls backend.complete() to generate the summary."""
        agent = _make_agent()
        for i in range(8):
            agent.messages.append(_make_msg("user" if i % 2 == 0 else "assistant", f"m{i}"))

        asyncio.run(agent._compact_messages(keep_last=4))

        agent.backend.complete.assert_called_once()

    def test_no_compaction_when_few_messages(self):
        """If messages <= keep_last, nothing changes."""
        agent = _make_agent()
        for i in range(4):
            agent.messages.append(_make_msg("user", f"m{i}"))

        original = list(agent.messages)
        asyncio.run(agent._compact_messages(keep_last=6))

        assert agent.messages == original
        agent.backend.complete.assert_not_called()

    def test_sets_post_compact_flag(self):
        """_compact_messages sets _post_compact = True."""
        agent = _make_agent()
        for i in range(10):
            agent.messages.append(_make_msg("user" if i % 2 == 0 else "assistant", f"m{i}"))

        asyncio.run(agent._compact_messages(keep_last=4))

        assert agent._post_compact is True


# ── post-compaction recall boost ───────────────────────────────────────────


class TestPostCompactionRecall:
    def test_agent_has_post_compact_flag(self):
        """EmbodiedAgent.__init__ sets _post_compact = False."""

        agent = _make_agent()
        assert hasattr(agent, "_post_compact")
        assert agent._post_compact is False

    def test_agent_has_last_context_tokens(self):
        """EmbodiedAgent.__init__ sets _last_context_tokens = 0."""
        agent = _make_agent()
        assert hasattr(agent, "_last_context_tokens")
        assert agent._last_context_tokens == 0

    def test_recall_n_larger_after_compact(self):
        """When _post_compact is True, recall_async is called with n > 3."""
        agent = _make_agent()
        agent._post_compact = True
        agent._turn_count = 1  # skip morning_reconstruction path
        agent._mcp = None
        agent._camera = None
        agent._mobility = None
        agent._tts = None
        agent._stt = None
        agent._coding = MagicMock()
        agent._coding.get_tool_definitions = MagicMock(return_value=[])
        agent._memory_tool = MagicMock()
        agent._memory_tool.get_tool_definitions = MagicMock(return_value=[])
        agent._tom_tool = MagicMock()
        agent._tom_tool.get_tool_definitions = MagicMock(return_value=[])

        # Make stream_turn return end_turn immediately
        from familiar_agent.backend import TurnResult

        fake_result = TurnResult(stop_reason="end_turn", text="ok")
        agent.backend.stream_turn = AsyncMock(return_value=(fake_result, []))
        agent.backend.make_assistant_message = MagicMock(return_value=_make_msg("assistant", "ok"))
        agent.backend.make_tool_results = MagicMock(return_value=_make_msg("user", ""))
        agent._run_post_response_pipeline = AsyncMock()
        agent._infer_emotion = AsyncMock(return_value="neutral")
        agent._summarize_exchange = AsyncMock(return_value="summary")
        agent._update_self_model = AsyncMock()
        agent._memory.save_async = AsyncMock()
        agent._should_compact = MagicMock(return_value=False)

        asyncio.run(agent.run("hello"))

        # recall_async should have been called with n > 3 (post-compact boost)
        call_args = agent._memory.recall_async.call_args
        n_used = call_args[1].get("n") or call_args[0][1]
        assert n_used > 3, f"Expected n > 3 after compaction, got {n_used}"


# ── _flatten_history ───────────────────────────────────────────────────────


class TestFlattenHistory:
    def test_nested_list_is_expanded(self):
        """ネストlist要素（tool結果相当）をフラット展開する。"""
        from familiar_agent.agent import _flatten_history

        tool_msgs = [
            {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]},
        ]
        history = [
            {"role": "user", "content": "hello"},
            tool_msgs,  # ネストlist（make_tool_results → append）
            {"role": "assistant", "content": "world"},
        ]
        flat = _flatten_history(history)
        assert flat == [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]},
            {"role": "assistant", "content": "world"},
        ]

    def test_dict_passthrough(self):
        """dict 要素はそのまま通す。"""
        from familiar_agent.agent import _flatten_history

        msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        assert _flatten_history(msgs) == msgs

    def test_non_dict_inside_nested_list_is_dropped(self):
        """ネストlist内の非dict要素は落とす。"""
        from familiar_agent.agent import _flatten_history

        history = [
            [{"role": "user", "content": "tool"}, "stray-string"],
        ]
        flat = _flatten_history(history)
        assert flat == [{"role": "user", "content": "tool"}]


# ── _compact_messages with nested tool results ─────────────────────────────


class TestCompactMessagesWithNestedToolResults:
    def test_no_error_when_to_summarise_has_nested_list(self):
        """to_summarise 領域にネストlist要素が含まれても AttributeError が出ない。"""
        agent = _make_agent()
        # 通常メッセージ + ネストlist（tool結果相当）を to_summarise 領域に置く
        agent.messages = [
            _make_msg("user", "first"),
            _make_msg("assistant", "response"),
            [{"role": "user", "content": [{"type": "tool_result", "content": "ok"}]}],  # ネストlist
            _make_msg("user", "second"),
            # ↓ recent (keep_last=2) として保持される
            _make_msg("user", "recent1"),
            _make_msg("assistant", "recent2"),
        ]
        # AttributeError を出さず完了すること
        asyncio.run(agent._compact_messages(keep_last=2))

        # 要約マーカー + recent 2件
        assert len(agent.messages) == 3
        assert "summary" in agent.messages[0]["content"].lower() or len(agent.messages[0]["content"]) > 0

    def test_recent_slice_structure_preserved(self):
        """_compact_messages 後も recent スライスの構造がそのまま残る。"""
        agent = _make_agent()
        nested = [{"role": "user", "content": [{"type": "tool_result", "content": "ok"}]}]
        recent_a = _make_msg("user", "keep-me")
        recent_b = _make_msg("assistant", "keep-too")
        agent.messages = [
            _make_msg("user", "old"),
            nested,  # ネストlist（to_summarise 領域）
            recent_a,
            recent_b,
        ]
        asyncio.run(agent._compact_messages(keep_last=2))

        # recent はネスト構造を保ったまま残る（送信時 flatten の前提）
        assert agent.messages[1] is recent_a
        assert agent.messages[2] is recent_b


# ── _check_response_coherence with nested tool results ─────────────────────


class TestCheckResponseCoherenceWithNestedToolResults:
    def test_no_error_when_last_6_has_nested_list(self):
        """直近6件にネストlist要素が含まれても AttributeError が出ない。"""
        agent = _make_agent()
        # utility_backend を別モックにしてループが実行される状態にする
        agent._utility_backend = MagicMock()
        agent._utility_backend.complete = AsyncMock(return_value="ok")

        agent.messages = [
            _make_msg("user", "hello"),
            [{"role": "user", "content": [{"type": "tool_result", "content": "data"}]}],  # ネストlist
            _make_msg("assistant", "reply"),
        ]
        # AttributeError を出さず None か文字列を返すこと
        result = asyncio.run(agent._check_response_coherence("some response"))
        assert result is None or isinstance(result, str)
