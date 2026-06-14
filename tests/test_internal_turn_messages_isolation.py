"""Tests for internal-turn messages isolation (turn_messages ローカル化).

テスト設計の原則:
  主検証テスト(test_messages_body_not_converted, test_convert_not_called)は
  「修正前コードの行 3634 で self.messages が Gemini変換される」実体を捕えて Red になる。
  ターン後の最終状態確認は finally 復元(3951-3954)が効くため Red にならない。
  バグの本質は「内的ターン中、stream_turn 呼び出し瞬間に self.messages 本体が
  Gemini形式になる」点なので、ターン中を捉える。

Red 対応一覧:
  test_messages_body_not_converted_when_stream_turn_called: 修正前=Red / 修正後=Green ★主検証
  test_convert_not_called_on_body:                          修正前=Red / 修正後=Green ★補助
  test_internal_turn_result_not_in_history:                 回帰防止のみ (単体ではRedにならない)
  test_normal_turn_appends_to_history:                      回帰防止のみ (単体ではRedにならない)
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from familiar_agent.backend import GeminiBackend, TurnResult
from familiar_agent.exploration import ExplorationTracker


# ---------------------------------------------------------------------------
# Shape check
# ---------------------------------------------------------------------------


def _is_anthropic_shape(messages: list) -> bool:
    """Anthropic形式(role+content dict)であれば True。

    Gemini形式は parts を持ち content を持たない、または要素が list になる。
    """
    if not messages:
        return True  # 空リストは形式として問題ない
    return all(
        isinstance(m, dict) and "content" in m and "parts" not in m
        for m in messages
    )


# ---------------------------------------------------------------------------
# TurnResult / stream_turn helpers
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

    separate_utility=True  → _utility_backend ≠ backend (MagicMock, non-Anthropic)
                             → _maybe_swap_internal_backend が swap し、line 3634 が走る
    separate_utility=False → _utility_backend == backend (同一参照)
                             → swap しない (通常ターン相当)
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
        # 別オブジェクトにすることで _maybe_swap_internal_backend が swap する
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
    """重いasync呼び出しを抑制するパッチリスト（各フィクスチャで独立生成）。"""
    return [
        patch("familiar_agent.agent.EmbodiedAgent._morning_reconstruction",
              new=AsyncMock(return_value="")),
        patch("familiar_agent.agent.EmbodiedAgent._infer_companion_mood",
              new=AsyncMock(return_value="engaged")),
        patch("familiar_agent.agent.EmbodiedAgent._infer_emotion",
              new=AsyncMock(return_value="neutral")),
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
    """utility ≠ main backend の内的ターン用エージェント。

    - separate_utility=True: _maybe_swap_internal_backend が swap する
    - desire_name="reflect" は is_social_desire=False → swap 対象
    - utility バックエンドは MagicMock (非 AnthropicBackend) → line 3634 の変換条件を満たす
    - メッセージを事前シードし、_is_anthropic_shape チェックが非空で機能するようにする
    """
    agent = _make_agent(separate_utility=True)
    # 事前メッセージをシード：変換前後で形式の差が明確に出る
    agent.messages = [{"role": "user", "content": "以前のメッセージ"}]

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
async def test_messages_body_not_converted_when_stream_turn_called(internal_agent):
    """【主検証・Red】stream_turn 呼び出し瞬間、self.messages 本体は Anthropic形式のまま。

    修正前に落ちる行:
      3634: self.messages = GeminiBackend.convert_messages_to_gemini_format(self.messages)
      → stream_turn 呼び出し時(3651)に本体が Gemini形式(parts を持つ dict)になる
      → _is_anthropic_shape(agent.messages) == False → assert が落ちる(Red)

    修正後:
      turn_messages ローカルを渡す → 本体は変換されず dict のまま → Green
    """
    agent = internal_agent
    seen: dict = {}

    async def capturing_stream_turn(*, system, messages, tools, max_tokens, on_text=None):
        # stream_turn 呼び出しの瞬間に self.messages 本体の形式を記録する
        seen["body_is_anthropic"] = _is_anthropic_shape(agent.messages)
        return _end_turn_result(), None

    # utility backend の stream_turn をキャプチャ関数に差し替え
    agent._utility_backend.stream_turn = capturing_stream_turn

    await agent.run(user_input="", inner_voice="内省中...", desire_name="reflect")

    assert seen.get("body_is_anthropic") is True, (
        "stream_turn 呼び出し時、self.messages 本体が Gemini形式に変換されていた"
        f"（messages={agent.messages[:2]}）"
    )


@pytest.mark.asyncio
async def test_convert_not_called_on_body(internal_agent, monkeypatch):
    """【補助・Red】agent が self.messages 本体に対して Gemini変換を呼ばないこと。

    修正前に落ちる行:
      3634: GeminiBackend.convert_messages_to_gemini_format(self.messages) を呼ぶ
      → スパイが呼び出しを記録 → calls が非空 → assert が落ちる(Red)

    修正後:
      3634 が消えるため convert_messages_to_gemini_format が呼ばれない → Green
    """
    calls: list = []
    orig = GeminiBackend.convert_messages_to_gemini_format

    def spy(messages):
        calls.append(messages)
        return orig(messages)  # 動作は保持しつつ呼び出しを記録

    monkeypatch.setattr(GeminiBackend, "convert_messages_to_gemini_format", spy)

    agent = internal_agent
    agent._utility_backend.stream_turn = AsyncMock(side_effect=_fake_end_turn_stream)

    await agent.run(user_input="", inner_voice="内省中...", desire_name="reflect")

    assert calls == [], (
        f"agent が self.messages 本体を Gemini変換した（{len(calls)}回呼ばれた）"
    )


@pytest.mark.asyncio
async def test_internal_turn_result_not_in_history(internal_agent):
    """【回帰防止】内的ターンの LLM 応答(assistant/tool)が self.messages に残らない。

    このテスト単体では Red にならない（修正前も finally 復元が働くため）。
    修正後は turn_messages がローカルなので構造的に保証される。
    """
    agent = internal_agent
    agent._utility_backend.stream_turn = AsyncMock(side_effect=_fake_end_turn_stream)

    await agent.run(user_input="", inner_voice="内省中...", desire_name="reflect")

    leaked = [m for m in agent.messages if isinstance(m, dict) and m.get("role") in {"assistant", "tool"}]
    assert leaked == [], f"内的ターンの LLM 応答が self.messages に漏れた: {leaked}"


@pytest.mark.asyncio
async def test_normal_turn_appends_to_history(normal_agent):
    """【回帰防止】通常ターンは self.messages に会話履歴が残る。

    修正後も turn_messages = self.messages（同一参照）なので append が本体に反映される。
    """
    agent = normal_agent
    n_before = len(agent.messages)
    agent.backend.stream_turn = AsyncMock(side_effect=_fake_end_turn_stream)

    await agent.run(user_input="こんにちは")

    assert len(agent.messages) > n_before, "通常ターンは履歴が増えるはず"
    assert _is_anthropic_shape(agent.messages), "通常ターン後も全要素は Anthropic形式のはず"
