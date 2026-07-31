"""相手の気分の判定（`_infer_companion_mood`）。

相手の文面から気分ラベルを決める。判定した気分を内受容テキストへ差し込む経路は、
旧 ReAct のプロンプト組み立てごと撤去した（実行中のプロンプトは
`build_event_system_prompt` が組む）。ここに残るのは判定そのもののテストである。
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

class TestInferCompanionMood:
    """_infer_companion_mood() returns a valid mood label from LLM backend."""

    def _make_agent_with_mock_backend(self, complete_return: str):
        """Create an EmbodiedAgent with a mock backend that returns complete_return."""
        from familiar_agent.agent import EmbodiedAgent
        from familiar_agent.config import AgentConfig

        config = AgentConfig.__new__(AgentConfig)
        config.camera = MagicMock()
        config.camera.host = None
        config.mobility = MagicMock()
        config.mobility.api_key = None
        config.tts = MagicMock()
        config.tts.elevenlabs_api_key = None
        config.stt = MagicMock()
        config.stt.elevenlabs_api_key = None
        config.coding = MagicMock()
        config.coding.enabled = False
        config.max_tokens = 1024
        config.companion_name = "Kouta"

        agent = EmbodiedAgent.__new__(EmbodiedAgent)
        agent.config = config
        agent.messages = []
        agent._started_at = time.time()
        agent._turn_count = 0
        agent._me_md = ""
        agent._camera = None
        agent._mobility = None
        agent._tts = None
        agent._stt = None
        agent._mcp = None
        agent._session_input_tokens = 0
        agent._session_output_tokens = 0
        agent._last_context_tokens = 0
        agent._post_compact = False
        agent._background_tasks = set()
        agent._cached_plan_ctx = ""
        agent._cached_workspace_ctx = ""
        agent._cached_temporal_ctx = None
        agent._cached_companion_mood = "engaged"

        from familiar_agent.tools.memory import ObservationMemory, MemoryTool
        from familiar_agent.tools.coding import CodingTool

        agent._memory = MagicMock(spec=ObservationMemory)
        agent._memory.recall_day_summaries_async = AsyncMock(return_value=[])
        agent._memory.recall_semantic_facts_async = AsyncMock(return_value=[])
        agent._memory.recall_behavior_policies_async = AsyncMock(return_value=[])
        agent._memory.format_semantic_facts_for_context = MagicMock(return_value="")
        agent._memory.format_behavior_policies_for_context = MagicMock(return_value="")
        agent._memory_tool = MagicMock(spec=MemoryTool)
        agent._coding = MagicMock(spec=CodingTool)

        from familiar_agent.exploration import ExplorationTracker
        from familiar_agent.self_narrative import SelfNarrative
        from familiar_agent.relationship import RelationshipTracker
        from familiar_agent.prediction import PredictionEngine

        agent._exploration = ExplorationTracker()
        agent._scene = None
        agent._self_narrative = SelfNarrative()
        agent._relationship = RelationshipTracker()
        agent._prediction = PredictionEngine()
        agent._memory.as_coalition_async = AsyncMock(return_value=None)
        agent._memory_worker = MagicMock()
        agent._memory_worker.is_running = True
        agent._mood = "neutral"
        agent._mood_intensity = 0.0
        agent._mood_set_at = time.time()

        mock_backend = MagicMock()
        mock_backend.complete = AsyncMock(return_value=complete_return)
        agent.backend = mock_backend
        agent._utility_backend = mock_backend

        return agent

    @pytest.mark.asyncio
    async def test_returns_engaged(self):
        agent = self._make_agent_with_mock_backend("engaged")
        result = await agent._infer_companion_mood("Let's work on this together!")
        assert result == "engaged"

    @pytest.mark.asyncio
    async def test_returns_tired(self):
        agent = self._make_agent_with_mock_backend("tired")
        result = await agent._infer_companion_mood("I'm so tired today...")
        assert result == "tired"

    @pytest.mark.asyncio
    async def test_returns_frustrated(self):
        agent = self._make_agent_with_mock_backend("frustrated")
        result = await agent._infer_companion_mood("This isn't working at all!")
        assert result == "frustrated"

    @pytest.mark.asyncio
    async def test_returns_absent(self):
        agent = self._make_agent_with_mock_backend("absent")
        result = await agent._infer_companion_mood("...")
        assert result == "absent"

    @pytest.mark.asyncio
    async def test_returns_happy(self):
        agent = self._make_agent_with_mock_backend("happy")
        result = await agent._infer_companion_mood("That worked perfectly!")
        assert result == "happy"

    @pytest.mark.asyncio
    async def test_empty_string_returns_absent(self):
        """Empty input → absent without calling backend."""
        agent = self._make_agent_with_mock_backend("happy")
        result = await agent._infer_companion_mood("")
        assert result == "absent"
        agent.backend.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_only_returns_absent(self):
        agent = self._make_agent_with_mock_backend("happy")
        result = await agent._infer_companion_mood("   ")
        assert result == "absent"
        agent.backend.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_very_short_returns_absent(self):
        """Messages shorter than 3 chars → absent."""
        agent = self._make_agent_with_mock_backend("happy")
        result = await agent._infer_companion_mood("ok")
        assert result == "absent"

    @pytest.mark.asyncio
    async def test_invalid_backend_response_falls_back_to_engaged(self):
        """If backend returns garbage, fall back to 'engaged'."""
        agent = self._make_agent_with_mock_backend("UNKNOWN_LABEL_XYZ")
        result = await agent._infer_companion_mood("Hello there friend!")
        assert result == "engaged"

    @pytest.mark.asyncio
    async def test_label_stripped_and_lowercased(self):
        """Backend returning '  Tired  ' (with spaces/caps) is normalised."""
        agent = self._make_agent_with_mock_backend("  Tired  ")
        result = await agent._infer_companion_mood("I'm exhausted")
        assert result == "tired"

