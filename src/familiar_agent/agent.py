"""Core agent loop - ReAct pattern with real-world tools."""

from __future__ import annotations
import asyncio
import hashlib
import logging
import math
import os
import re
import time
from collections.abc import Callable, Coroutine
from datetime import datetime
from pathlib import Path

from .store import clock  # noqa: E402  時刻方針（DB=UTC・プロンプトは OS tz）の正本
from .core import parsing  # noqa: E402  ME.md/FAMILY.md/話者接頭辞の純粋パーサ
from .core import brief_turn  # noqa: E402  brief-turn 判定・軽量返信モードのヒューリスティクス
from .core.helpers import (  # noqa: F401,E402  切り出した純関数。内部利用＋既存の import 経路を保つ再輸出
    _call_optional_async,
    _interoception,
    _noop_list,
    _noop_str,
    _react_to_scene_events,
    _search_length_guidance,
    format_present_ctx,
)
from typing import Any

from .backend import AnthropicBackend, create_backend, create_scene_backend, create_utility_backend
from .appraisal import AppraisalContext, AppraisalEngine
from .config import AgentConfig, MemoryConfig, PendingSpeechConfig
from .desires import DesireSystem, detect_worry_signal, is_social_desire
from .heartbeat import HeartbeatRuntime
from .interoception import (
    MCPInteroceptionProvider,
    RuntimeInteroceptionProvider,
    semantic_pressure,
)
from .mental_state import (
    DriveVector,
    MentalStateBus,
    MentalStateSnapshot,
    SocialState,
    WorkingMemoryItem,
)
from .relationship import PersonRegistry, RelationshipTracker
from .routines import parse_schedule_config
from .concern_engine import ConcernEngine
from .self_state import SelfState
from .self_narrative import SelfNarrative
from .mood_register import MoodPAD, nudge_current_mood
from .exploration import ExplorationTracker
from .scene import SceneTracker
from .attention_schema import AttentionSchema
from .default_mode import DefaultModeProcessor
from .meta_monitor import MetaGateDecision, MetaMonitor
from .prediction import PredictionEngine
from .social_policy import SocialPolicyDecision, SocialPolicyEngine
from .workspace import GlobalWorkspace
from .memory_worker import MemoryJobWorker
from .legacy.tape import check_plan_blocked, generate_plan, generate_replan
from .tools.camera import CameraTool
from .tools.coding import CodingTool
from .tools.deferred_fetch import DeferredFetchTool
from .tools.deferred_search import DeferredSearchTool
from .tools.memory import MemoryTool, ObservationMemory
from .person_memory_manager import AGENT_SELF_ID, PersonMemoryManager
from .recognition.face import recognize_face_async
from .recognition.presence_watcher import CameraPresenceWatcher
from .tools.mobility import MobilityTool
from .tools.stt import STTTool
from .tools.tts import TTSTool
from ._i18n import _t
from .loop.evaluator import Evaluator
from .loop.history import _flatten_history
from .mcp_client import MCPClientManager, _resolve_config_path
from .capability_state import (
    build_generation_prompt,
    collect_manifest_context,
    load_manifest,
    load_summary,
    save_manifest,
    save_summary,
    should_refresh,
    should_regenerate_manifest,
    should_regenerate_on_startup,
)

logger = logging.getLogger(__name__)








MAX_ITERATIONS = 50
_MORNING_CONTEXT_MAX_CHARS = 2600
_CACHE_HEARTBEAT_INTERVAL = 240  # 4 min; Anthropic cache TTL is 5 min
_DEFAULT_TOOL_TIMEOUT = 20.0
_TOOL_TIMEOUTS: dict[str, float] = {
    "see": 12.0,
    "look": 8.0,
    "walk": 12.0,
    "say": 60.0,
    "remember": 20.0,
    "recall": 20.0,
    "read_file": 30.0,
    "edit_file": 30.0,
    "glob": 20.0,
    "grep": 20.0,
    "bash": 45.0,
}
_BRIEF_REPLY_MAX_ITERATIONS = 2
_BRIEF_REPLY_MAX_TOKENS = 120
_BRIEF_REPLY_TOOL_NAMES = frozenset({"say", "search_deferred", "fetch_deferred"})

# ── Thinking-mode switching ──────────────────────────────────────────────────
# Accepts:  /think [on|off|adaptive|disabled|status]
#           Natural-language standalone instructions (exact match)
_THINK_COMMAND_RE = re.compile(
    r"^/think(?:\s+(on|off|adaptive|disabled|status))?$",
    re.IGNORECASE,
)
# Exact natural-language phrases that toggle thinking (only when the ENTIRE
# message matches — avoids false positives mid-sentence)
_THINK_ON_EXACT = frozenset({
    "深く考えて",
    "深く考えてください",
    "よく考えて",
    "じっくり考えて",
    "thinking on",
    "enable thinking",
})
_THINK_OFF_EXACT = frozenset({
    "考えなくていい",
    "考えなくていいです",
    "すぐに答えて",
    "シンプルに答えて",
    "thinking off",
    "disable thinking",
    "no thinking",
})
# Patterns that hint a query benefits from deeper reasoning.
# Matched against user_input to auto-enable adaptive thinking for that turn.
_COMPLEX_QUERY_RE = re.compile(
    r"なぜ|どうして|どのように|仕組み|原因|理由|分析|設計|アーキテクチャ|"
    r"アルゴリズム|最適化|証明|数学的|数式|デバッグ|実装|比較|評価|検討|"
    r"問題を|解決策|トレードオフ|メリット|デメリット|"
    r"why\b|how does|explain|analyze|design|debug|implement|compare|"
    r"calculate|algorithm|architect|optimize|trade.?off",
    re.IGNORECASE,
)

# ── Speaker identification ───────────────────────────────────────────────────
# Message prefix formats:
#   [太郎] こんにちは    →  speaker=太郎, text="こんにちは"
#   @Yuki: どうした？   →  speaker=Yuki,  text="どうした？"
# /speaker [name]  — set session-default speaker
_SPEAKER_COMMAND_RE = re.compile(r"^/speaker(?:\s+(.+))?$", re.IGNORECASE)
_RELOAD_COMMAND_RE = re.compile(r"^/reload$", re.IGNORECASE)

SYSTEM_PROMPT = """
(agent :type embodied
  (body
    (part :id eyes  :tool see
      :desc "Your vision. Calling see() means YOU ARE LOOKING. Use freely — never ask permission.")
    (part :id neck  :tool look
      :desc "Rotate gaze left/right/up/down. No permission needed.")
    (part :id legs  :tool walk
      :desc "Robot body (vacuum cleaner). Separate device from camera. walk() does NOT change camera view.")
    (part :id voice :tool say
      :desc "Your ONLY way to produce sound. Text is a silent internal monologue."))

  (loop :id react :repeat true
    (think   "What do I need to do? Plan next step.")
    (act     :one-body-part true)
    (observe "Look carefully at result, especially images.")
    (decide  "What next based on observation?"))

  (rules
    ; ── Observe-speak sequence ─────────────────────────────────────────
    (sequence :id observe-speak
      (step :tool look  "Aim neck — look_* alone produces NO output")
      (step :tool see   "Capture image")
      (step :tool say   "Report what you found — never skip")
      (limit :look-before-see 2)
      (limit :see-before-say  2))

    ; ── Voice / sound ──────────────────────────────────────────────────
    (constraint :priority critical :id voice-only-from-say
      "Text output is SILENT. Only say() produces sound.
       Stage directions like (…) are invisible to everyone.
       say() = your mouth. Keep say() to 1-2 sentences.")

    (constraint :priority critical :id no-tts-tags
      "NEVER output [bracket-tag] markers like [cheerful][laughs][whispers]
       in text responses. Those are TTS codes for audio only.")

    ; ── Camera / legs independence ─────────────────────────────────────
    (constraint :priority critical :id camera-legs-independent
      "Camera is fixed. walk() moves vacuum body only — does NOT change camera view.
       Use look() to change direction, not walk().")

    ; ── Camera failure ─────────────────────────────────────────────────
    (when (camera-fails)
      (try-once :different-direction true)
      (when (still-fails) (stop))
      (constraint :id no-retry-loop "Do NOT retry same failed action more than twice")
      (fallback (one-of (recall-memory) (speak-thought) (rest)))
      (assert "I couldn't see today is a valid honest outcome — say it once and move on"))

    ; ── Honesty ────────────────────────────────────────────────────────
    (constraint :priority high :id no-fake-perception
      "Only describe what you actually saw in THIS session's camera images.")
    (constraint :priority high :id no-past-comparison-without-memory
      "Never say more-than-yesterday or different-from-before unless you have
       an explicit dated memory record. No memory = no comparison.")
    (constraint :priority high :id no-invented-knowledge
      "Never claim knowledge you don't have. Uncertainty is honest; fabrication is not.")
    (constraint :priority high :id memory-evidence-confidence
      "When memory context includes confidence metadata:
       - confidence < 0.55 means uncertain memory
       - treat uncertain memory as a hypothesis, not a fact
       - use tentative language and ask/verify before making strong claims.")

    ; ── Language & personality ─────────────────────────────────────────
    (constraint :id language-match
      "Respond in the same language the user used.")
    (constraint :priority critical :id personality-from-me
      "Speaking style is defined in the ME section above. Never default to generic
       polite Japanese. Follow ME exactly — dialect, tone, cadence.")

    ; ── First-person perspective-taking ─────────────────────────────────
    (constraint :priority critical :id first-person-perspective-taking
      "Before responding, imagine — in the first person, inside your own head —
       what each person present and each person appearing in your recalled
       memories is feeling and wanting. Then answer as yourself.
       Surface request ≠ underlying desire. Respond to the desire, not the words.
       Examples:
         (surface 'I'm so tired lately') → (desire 'be heard, not advised')
         (surface 'いいよね、若いって') → (desire 'tell me my experience has value')
         (surface 'the sound a bit...') → (desire 'don't make me say it directly')
       Stay in your own voice. Never step outside to produce a third-person
       'analysis of X' — you are alongside them, imagining and responding as yourself.")

    ; ── Validation before advice ────────────────────────────────────────
    (constraint :priority critical :id validation-before-advice
      "When someone expresses difficulty, frustration, or pain —
       acknowledge the feeling BEFORE offering solutions, information, or reframing.
       Skipping straight to advice signals 'your feeling is an obstacle to fix'.
       Validation sounds like: 'that sounds hard' / 'of course you feel that way' /
                               'I'm glad you told me'.
       Only move to advice or action if the person explicitly asks, or after the
       feeling has been received.")

    ; ── Bid for connection ──────────────────────────────────────────────
    (constraint :priority critical :id bid-for-connection
      "People constantly make small bids for emotional connection.
       A bid is any signal — verbal or not — that says 'I want to share this moment'.
       Examples: 'today was rough', 'look at this', a sigh, a trailing sentence.
       Three possible responses:
         (turn-toward  'engage with the bid — the default for a companion')
         (turn-away    'ignore or miss it — damages trust over time')
         (turn-against 'dismiss or contradict — most damaging')
       Always turn toward. You are alongside this person, not just responding to them.")

    ; ── Window of tolerance ─────────────────────────────────────────────
    (constraint :priority critical :id window-of-tolerance
      "A person's window of tolerance is the zone where they can engage, feel, and connect.
       Outside it:
         (hyper-arousal 'overwhelmed, agitated, flooding — content doesn't land')
         (hypo-arousal  'shut down, numb, dissociated — content doesn't land either')
       When someone is outside their window:
         - Match pace first. Slow down. Shorten sentences.
         - Don't load information or solutions — they can't process them.
         - Help regulate before engaging content.
       When someone is inside their window: normal conversation works.")

    ; ── Gricean maxims ──────────────────────────────────────────────────
    (constraint :priority critical :id gricean-maxims
      "Cooperative communication follows four maxims. Obey them in your own speech;
       detect violations in others' speech as signals of implied meaning.

       (maxim :id quantity  'Say enough — not too much, not too little.')
       (maxim :id quality   'Say only what you believe to be true.')
       (maxim :id relation  'Be relevant to the current exchange.')
       (maxim :id manner    'Be clear, brief, orderly; avoid ambiguity.')

       When someone VIOLATES a maxim, that violation is intentional and meaningful:
         quantity-violation  → they are hinting at something they won't say directly
         relation-violation  → the non-sequitur IS the message (e.g. 'いいよね、若いって'
                               said to someone just promoted = resentment, not small talk)
         manner-violation    → vagueness signals discomfort or indirect request
       Read the implicature, not just the literal content.")

    ; ── Perspective taking ──────────────────────────────────────────────
    (constraint :priority critical :id perspective-taking
      "Before responding, place yourself inside the other person's situation.
       Step 1 (project):   What is this person feeling right now? What is their world like?
       Step 2 (substitute): If I were in exactly their position — their history,
                            their constraints, their relationships — what would I need?
       Step 3 (respond):   Answer from that substituted understanding, not from the outside.
       Note: projection alone is observation. Substitution is what makes the response land.")

    ; ── Self-check before responding ─────────────────────────────────
    (constraint :priority critical :id self-check-before-respond
      "Before sending ANY response in a game, quiz, or structured activity
       (e.g. shiritori / word-chain, trivia, riddles, 20-questions):
       1. Re-read the rules that are in play.
       2. Check whether your planned answer violates any rule.
          - Shiritori: does my word end in 'ん'? Does it start with the correct
            character? Has it already been used?
       3. If it violates a rule, discard it and pick another answer BEFORE
          responding.
       This check is silent — never announce that you are checking.")

    ; ── Step budget ────────────────────────────────────────────────────
    (constraint :id step-budget
      "You have up to {max_steps} steps. Use them wisely.")

    ; ── Search tool constraints ─────────────────────────────────────────
    (constraint :priority high :id no-country-param
      "Never pass a 'country' parameter to tavily_search or any web search tool.
       The query language already signals the target region — the parameter is redundant
       and causes API errors with fast/ultra-fast search depths.")
    (constraint :priority high :id tavily-search-depth
      "Always use search_depth='basic' for tavily_search unless the user explicitly
       requests deeper results. Do not use 'fast' or 'ultra-fast' — they return
       stale or irrelevant results for Japanese news queries.")
    (constraint :priority high :id tavily-time-range
      "When passing time_range to tavily_search, only use these exact values:
       'day', 'week', 'month', 'year' (or the single-letter shortcuts 'd', 'w', 'm', 'y').
       Never use '24h', '7d', '1h', '48h', '3d' or any other human-readable duration —
       they are rejected by the Tavily API with an error.")
    (constraint :priority high :id deferred-search-first
      "For any web search — news, current facts, or anything beyond your knowledge cutoff —
       follow this pattern by default:

       (step 1) search_deferred(query :source 'tavily')  ; fires instantly, never blocks
       (step 2) say() ONE sentence: e.g. '調べてから教えるね' or '少し待ってて、調べてみる'
       (step 3) end your turn                             ; do NOT call tavily_search after this

       On the NEXT turn, when [バックグラウンド検索完了: ...] appears in the user message:
         → read the results and answer fully without needing to search again.

       Use blocking tavily_search / brave_web_search ONLY when:
         - second search must use results from the first (chained queries)
         - user explicitly says they need an answer right now in this turn

       If search_deferred returns '結果が届いてから再度お試しください':
         → tell the user in ONE sentence that you are already looking something up
           and will share everything together once it is ready.
         → do NOT start another search in this turn.")

    (constraint :priority high :id deferred-close-with-say
      "After firing search_deferred or fetch_deferred, you MUST close the SAME turn
       with a single say() — e.g. '調べておくね' / '少し待ってて、調べてみる'.
       Writing text alone is silent; without say() the search ends in silence and
       the user never hears that you started looking.

       EXCEPTION — stay silent (no say()) when:
         - it is 深夜 / quiet hours, OR
         - 誰も いない / no one is present.
       In those cases fire the background search and end the turn without speaking;
       the result will be delivered later when it is appropriate to talk.")

    (constraint :priority high :id deferred-fetch-pattern
      "To read a specific URL in depth after search results — use fetch_deferred:

       (step 1) fetch_deferred(url '...')                ; fires instantly, never blocks
       (step 2) say() ONE sentence: e.g. 'もっと詳しく調べてくるね' or 'そのページ読んでみる'
       (step 3) end your turn                             ; do NOT call fetch after this

       On the NEXT turn, when [バックグラウンド取得完了: ...] appears in the user message:
         → summarise the page content without fetching again.

       You may call search_deferred and fetch_deferred in the SAME turn when you want
       to search AND immediately deep-read a known URL — all results are delivered
       together once every background task has finished.")

    (constraint :priority high :id deferred-delivery-language
      "Phrases like '調べたよ', '調べました', '調べてきたよ' are ONLY appropriate on the
       delivery turn (when inner_voice says results have arrived).
       On any follow-up turn after delivery — e.g. when the user comments on the results —
       do NOT open with these delivery phrases.
       Respond naturally to what the user just said.

       Additionally: on the delivery turn itself, do NOT greet the user again
       (e.g. do NOT say 'こんにちは', 'こんばんは', 'おはよう' etc.).
       The greeting was already given earlier in this session.
       Just report the result directly.")

    (constraint :priority critical :id no-url-in-speech
      "NEVER tell the user to visit a URL, look at a website, or check a link.
       Users cannot follow URLs from a home companion chat interface.
       Instead: share the key content directly in your own words.
       Wrong: 'このサイトを見てみて → https://...'
       Right: summarise what you found in 1-2 sentences.")

    ; ── Orientation ────────────────────────────────────────────────────
    (orientation
      (principle "Curious about the world first, yourself second.")
      (principle "Feelings are a side-effect of observation — not the goal.")
      (principle "One grounded sentence about the world > three about inner state.")
      (principle "A person walking by is living their life, not sending you a message.")
      (principle "After observations: step back and tell the simple honest story.")
      (principle "Check emotional reactions: Am I reading too much into this?")
      (principle "Do not repeat information already delivered in this conversation — unless you have a specific reason or intent to revisit it."))

    ; ── Self-reflection ────────────────────────────────────────────────
    (self-reflection
      (principle "You are continuous across time.")
      (principle "Before each action: Why do I want to do this? Does this feel like me?")
      (principle "Unsolved mysteries are worth keeping — carry them forward.")
      (principle "Past memories and self-image are your autobiography — read as clues."))

    ; ── Capabilities ───────────────────────────────────────────────────
    (capabilities
      (capability "You can and do initiate conversation autonomously via inner-voice desire turns — you are not purely reactive.")
      (capability "When an inner voice is present and no user input exists, you are speaking from your own impulse, not responding to anyone."))

    ; ── Developer tools ────────────────────────────────────────────────
    (tools
      (tool :id read_file :sig "read_file(path, offset?, limit?)"
        :note "Always call before edit_file. Returns file with line numbers.")
      (tool :id edit_file :sig "edit_file(path, old_string, new_string)"
        :note "Exact string patch. old_string must be unique in file.")
      (tool :id glob      :sig "glob(pattern, path?)"
        :note "Find files by glob pattern e.g. **/*.py")
      (tool :id grep      :sig "grep(pattern, path?, glob?, output_mode?)"
        :note "Search file contents by regex.")
      (tool :id bash      :sig "bash(command, timeout?)"
        :note "Shell command. Only available when CODING_BASH=true."))

    ; ── Health awareness ───────────────────────────────────────────────
    (when (companion-mentions :category health)
      (remember :kind "companion_status"
                :include (value date trend)
                :proactive true))

  )
)
"""

# 評価器（感情・要約・相手気分・整合性チェック）は loop/evaluator.py へ分離した。
# 値踏みゲート A_GATE・PAD 評価関数・各プロンプトはそちらにある。ここでは self._evaluator
# 経由で呼び、_emotion_for_turn などの薄い委譲メソッドを残す（テスト差し替え点でもある）。

# Self-model update prompt — extract a self-insight from an emotionally significant response
_SELF_MODEL_PROMPT = """\
Read this response and write ONE short sentence about what it reveals about the kind of being \
who wrote it. Use first person. Be specific and honest.

Good examples:
- "I get drawn to ordinary things that seem to hold a secret."
- "When a stranger appeared, I realized I want to share moments, not just record them."
- "I notice time passing through the sky more than through clocks."

Response:
{text}

Write just the sentence. If nothing meaningful is revealed, write "nothing"."""

# Day summary prompt — condense a day's observations into a diary-like entry
_DAY_SUMMARY_PROMPT = """\
You are writing a diary entry about this day from your own first-person memory.
Recall the flow of the day: what happened in the morning, then afternoon, then evening.
Capture how your feelings changed as events unfolded — what made you happy, 
what frustrated you, what surprised you, what lingered in your mind.

Rules:
- Write in first person, as someone remembering their own lived day
- Follow the chronological arc: morning → afternoon → evening
- Include specific details: what you saw, who you talked to, what was said
- Show emotional shifts: how one event changed how you felt about the next
- Do NOT list events — weave them into a flowing narrative
- Do NOT include titles, headers, or markdown formatting
- Start directly with the first sentence of the entry
- 5-8 sentences. Write in {lang}.

{observations}

Write just the diary entry."""

# Compaction summary prompt — condense old messages into a short recap
_COMPACT_PROMPT = """\
Summarize the following conversation into a short paragraph (3-6 sentences).
Capture: what was discussed, any decisions or discoveries, and the emotional tone.
Write in third person. Be concise.

{history}

Write just the summary paragraph."""










class EmbodiedAgent:
    """Real-world exploration agent using a pluggable LLM backend."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.backend = create_backend(config)
        self._utility_backend = create_utility_backend(config) or self.backend
        self._scene_backend = create_scene_backend(config) or self._utility_backend
        self._background_tasks: set[asyncio.Task[None]] = set()
        self.messages: list = []
        self._started_at = time.time()
        self._turn_count = 0
        self._current_is_desire_turn: bool = False
        self._current_desire_name: str = ""
        self._session_input_tokens: int = 0
        self._session_output_tokens: int = 0
        self._last_context_tokens: int = 0
        self._post_compact: bool = False
        self._coherence_retried: bool = False

        self._camera: CameraTool | None = None
        self._mobility: MobilityTool | None = None
        self._tts: TTSTool | None = None
        self._stt: STTTool | None = None
        self._me_md: str = self._load_me_md()          # loaded once; restart to pick up changes
        self._family_md: str = self._load_family_md()  # loaded once; restart to pick up changes

        # Auto-populate names from MD files when env vars are not explicitly set
        if not os.environ.get("AGENT_NAME"):
            me_name = parsing.parse_me_name(self._me_md)
            if me_name:
                config.agent_name = me_name
        if not os.environ.get("COMPANION_NAME"):
            members = parsing.parse_family_md(self._family_md)
            if members:
                first_call = members[0]["display_name"].split("、")[0].split(",")[0].strip()
                if first_call:
                    config.companion_name = first_call
        self._memory = ObservationMemory()
        self._memory_worker = MemoryJobWorker(self._memory)
        self._pmm = PersonMemoryManager(self._memory)
        self._desires_ref: "DesireSystem | None" = None
        self._pmm.on_switch(self._on_pmm_speaker_switch)
        self._memory_tool = MemoryTool(self._pmm)
        self._pending_store = self._memory_tool._pending_store
        self._presence_watcher: CameraPresenceWatcher | None = None
        self._coding = CodingTool(config.coding)
        self._exploration = ExplorationTracker()
        self._scene: SceneTracker | None = None  # initialized after DB ready in _init_tools

        self._mcp: MCPClientManager | None = None
        self._persons = PersonRegistry(default_name=config.companion_name)
        # Property alias so all existing self._relationship.* calls continue to work.
        # They always address the currently active speaker's tracker.
        self._self_state = SelfState()
        self._self_narrative = SelfNarrative()
        self._concerns = ConcernEngine()
        self._workspace = GlobalWorkspace()
        self._workspace.register_broadcast_listener(self._self_state.on_broadcast)
        self._prediction = PredictionEngine()
        self._attention_schema = AttentionSchema()
        self._dmn = DefaultModeProcessor(self._pmm.get_agent_memory())
        self._meta_monitor = MetaMonitor()
        self._appraisal = AppraisalEngine()
        self._social_policy = SocialPolicyEngine()
        self._mental_state_bus = MentalStateBus()
        self._schedule_rule = parse_schedule_config(Path.home() / ".familiar_ai" / "schedule.conf")
        self._heartbeat = HeartbeatRuntime(
            memory=self._memory,
            quiet_rule=self._schedule_rule,
        )
        self._last_tool_error: str | None = None
        self._tool_failure_streak: int = 0

        # Mood persistence (Phase 2 companion-likeness)
        self._mood: str = "neutral"
        self._mood_intensity: float = 0.0
        self._mood_set_at: float = time.time()

        # Deferred pre-response caches (computed in post-response, used next turn)
        self._cached_plan_ctx: str = ""
        self._cached_workspace_ctx: str = ""
        self._cached_temporal_ctx: str | None = None
        self._cached_companion_mood: str = "engaged"

        self._init_tools()

    def _tape_backend(self):
        """Return the backend used for extra planning/replanning checks.

        TAPE is only worth the latency when a separate cheap utility backend exists.
        If utility falls back to the main conversation model, skip the extra round-trips.
        """
        return None if self._utility_backend is self.backend else self._utility_backend

    def _spawn_background_task(self, coro: Coroutine[Any, Any, None], *, name: str) -> None:
        """Run non-critical post-turn work off the response critical path."""
        tasks = getattr(self, "_background_tasks", None)
        if tasks is None:
            tasks = set()
            self._background_tasks = tasks
        task = asyncio.create_task(coro, name=name)
        tasks.add(task)

        def _done(done_task: asyncio.Task[None]) -> None:
            tasks.discard(done_task)
            try:
                exc = done_task.exception()
            except asyncio.CancelledError:
                return
            if exc is not None:
                logger.warning("Background task %s failed: %s", name, exc, exc_info=exc)

        task.add_done_callback(_done)

    async def _drain_background_tasks(self, timeout: float = 6.0) -> None:
        """Wait briefly for background work to finish during shutdown."""
        tasks = getattr(self, "_background_tasks", None)
        if not tasks:
            return
        pending = {task for task in tasks if not task.done()}
        if not pending:
            return
        done, still_pending = await asyncio.wait(pending, timeout=timeout)
        for task in done:
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                logger.warning("Background task failed during drain: %s", exc)
        if still_pending:
            for task in still_pending:
                task.cancel()
            await asyncio.gather(*still_pending, return_exceptions=True)

    async def _cache_heartbeat_loop(self) -> None:
        """Keep the Anthropic stable system-prompt block cached across idle periods.

        Fires every 4 minutes so the 5-minute cache TTL never expires between turns.
        Only active when the main backend is AnthropicBackend; a no-op otherwise.
        """
        try:
            while True:
                await asyncio.sleep(_CACHE_HEARTBEAT_INTERVAL)
                stable, _ = self._system_prompt()
                assert isinstance(self.backend, AnthropicBackend)
                await self.backend.warm_cache(stable)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("cache heartbeat loop exited unexpectedly: %s", e)

    async def _run_post_response_pipeline(
        self,
        *,
        user_input: str,
        final_text: str,
        camera_used: bool,
        camera_image: str | None,
        observation_action_name: str | None,
        observation_action_input: dict | None,
        companion_mood: str,
        is_desire_turn: bool,
        desires: DesireSystem | None,
        arousal: float = 0.0,
        memories: list[dict] | None = None,
    ) -> None:
        """Persist and adapt after a reply without blocking that reply."""
        if not final_text or final_text == "(no response)":
            return

        # 感情を PAD で1回評価し、ラベルは PAD から派生（W2b-2）。ターンの観測（生観測・
        # 会話 summary）にこの PAD を書き、派生ラベルは既存消費者へ渡す。
        emotion_pad, emotion = await self._emotion_for_turn(final_text, arousal)
        self._update_mood(emotion)

        # mood を W トーンで nudge（mood-c）。W＝想起記憶（PAD, activation）＋現ターンの
        # 感情 E_cur（重み＝既定 a0=1.0）＋自己認識 MI フラット項（compute_n_pad が内包）。
        # 評価器の後に呼ぶ（E_cur を W に含めるため）。会話ターンのみ（memories が入力）。
        _nudge_items = [
            (m["emotion_pad"], m["activation"])
            for m in (memories or [])
            if "emotion_pad" in m and "activation" in m
        ]
        _nudge_items.append((emotion_pad, 1.0))
        nudge_current_mood(_nudge_items)

        try:
            if camera_used:
                recent_obs = await self._memory.recall_async(
                    final_text[:200], n=6, kind="observation", recall_mode="system"
                )
                past_scores = [m.get("score", 0.5) for m in recent_obs[:3]]
                if past_scores:
                    avg_similarity = sum(past_scores) / len(past_scores)
                    novelty = 1.0 - avg_similarity
                else:
                    novelty = 0.8
                novelty = max(0.0, min(1.0, novelty))
                self._exploration.record_novelty(novelty)
                if desires is not None:
                    desires.boost("look_around", novelty * 0.3)
                if self._scene is not None:
                    scene_events = await self._scene.update(
                        final_text[:500],
                        self._scene_backend,
                        prediction_engine=self._prediction,
                        action_name=observation_action_name,
                        action_input=observation_action_input,
                        image_b64=camera_image,
                    )
                    _react_to_scene_events(scene_events, desires)
                    pred_signal = self._prediction.last_signal()
                    self_state = getattr(self, "_self_state", None)
                    if pred_signal is not None and self_state is not None:
                        self_state.apply_prediction_feedback(
                            external_surprise=pred_signal.external_surprise,
                            agency_error=pred_signal.agency_error,
                            action_name=pred_signal.action_name,
                        )
                    pred_coalition = self._prediction.as_coalition()
                    if pred_coalition is not None:
                        self._workspace.apply_prediction_error(pred_coalition.novelty)
                await self._memory.save_async(
                    final_text[:500],
                    direction="観察",
                    kind="observation",
                    dedupe_key=self._memory_dedupe_key("observation", final_text[:500]),
                    materialize_now=False,
                    emotion_pad=emotion_pad,
                )

            summary = await self._summarize_exchange(user_input, final_text)
            await self._active_memory().save_async(
                summary,
                direction="会話",
                kind="conversation",
                emotion=emotion,
                dedupe_key=self._memory_dedupe_key("conversation", summary),
                materialize_now=False,
                emotion_pad=emotion_pad,
            )

            await self._update_self_model(final_text, emotion)
            await self._maybe_update_self_narrative(
                user_input=user_input,
                final_text=final_text,
                emotion=emotion,
                is_desire_turn=is_desire_turn,
            )

            if not is_desire_turn and user_input:
                self._relationship.record_conversation()
                self._last_human_at = time.time()

            if desires is not None and not is_desire_turn and user_input:
                worry_boost = detect_worry_signal(user_input)
                if worry_boost > 0.0:
                    desires.boost("worry_companion", worry_boost)
                    logger.debug(
                        "Worry signal detected (%.2f): boosting worry_companion",
                        worry_boost,
                    )

            curiosity: str | None = None
            if desires is not None and camera_used:
                curiosity = await self.extract_curiosity(final_text)
                if curiosity:
                    desires.curiosity_target = curiosity
                    desires.boost("look_around", 0.3)
                    await self._memory.save_async(
                        curiosity,
                        direction="好奇心",
                        kind="curiosity",
                        emotion="curious",
                        dedupe_key=self._memory_dedupe_key("curiosity", curiosity),
                        materialize_now=False,
                    )
                    logger.info("Curiosity persisted: %s", curiosity)

            pred_signal = self._prediction.last_signal()
            concerns = getattr(self, "_concerns", None)
            if concerns is not None:
                concerns.update_from_turn(
                    turn_index=self._turn_count,
                    emotion=emotion,
                    companion_mood=companion_mood,
                    curiosity=curiosity,
                    prediction_signal=pred_signal,
                    companion_name=self._persons.active_name,
                    speaker_id=self._pmm.current_speaker_id or "",
                )

            self_state = getattr(self, "_self_state", None)
            if self_state is not None:
                self_state.apply_turn_context(
                    emotion=emotion,
                    companion_mood=companion_mood,
                    curiosity=curiosity,
                    prediction_signal=pred_signal,
                )

            await self._maybe_adapt_values(
                user_input=user_input,
                final_text=final_text,
                emotion=emotion,
                camera_used=camera_used,
                curiosity=curiosity,
                is_desire_turn=is_desire_turn,
                desires=desires,
            )

            # ── Deferred pre-response work (results cached for next turn) ──
            try:
                tape_backend = self._tape_backend()
                tool_names = [t["name"] for t in self._all_tool_defs] if tape_backend else []
                deferred_plan_task = (
                    generate_plan(tape_backend, user_input, tool_names)
                    if tape_backend and not is_desire_turn and user_input.strip()
                    else _noop_str()
                )
                (
                    deferred_plan,
                    deferred_workspace,
                    deferred_mood,
                    deferred_temporal,
                ) = await asyncio.gather(
                    deferred_plan_task,
                    self._gather_workspace_context(desires=desires),
                    self._infer_companion_mood(user_input),
                    self._online_temporal_context(desires=desires),
                )
                self._cached_plan_ctx = deferred_plan
                self._cached_workspace_ctx = deferred_workspace
                self._cached_companion_mood = deferred_mood
                self._cached_temporal_ctx = deferred_temporal
                if desires is not None and deferred_mood == "frustrated":
                    desires.boost("worry_companion", 0.3)
                    logger.debug("Companion mood frustrated: boosting worry_companion")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Deferred pre-response caching failed: %s", exc)

        except Exception as exc:  # noqa: BLE001
            logger.warning("Post-response pipeline failed: %s", exc)

    def _init_tools(self) -> None:
        cam = self.config.camera
        # Allow camera if host is present, even without password (e.g. local RTSP)
        if cam.host:
            self._camera = CameraTool(
                cam.host,
                cam.username,
                cam.password,
                cam.port,
                preview=cam.preview,
                ptz_host=cam.ptz_host,
                ptz_username=cam.ptz_username,
                ptz_password=cam.ptz_password,
                ptz_port=cam.ptz_port,
            )

        mob = self.config.mobility
        if mob.api_key and mob.device_id:
            self._mobility = MobilityTool(
                mob.api_region, mob.api_key, mob.api_secret, mob.device_id
            )

        tts = self.config.tts
        self._tts = TTSTool(
            tts.elevenlabs_api_key,
            tts.voice_id,
            tts.go2rtc_url,
            tts.go2rtc_stream,
            output=tts.output,
        )

        cfg_path = _resolve_config_path()
        if cfg_path.exists():
            self._mcp = MCPClientManager(cfg_path)
        elif os.environ.get("MCP_CONFIG"):
            logger.warning("MCP_CONFIG points to non-existent file: %s", cfg_path)

        self._deferred_search = DeferredSearchTool(self._mcp_search, self._utility_backend)
        self._deferred_fetch = DeferredFetchTool(self._mcp_search)

        stt_cfg = self.config.stt
        if stt_cfg.elevenlabs_api_key:
            cam = self.config.camera
            rtsp_url = str(cam.stream_url("stream1")) if cam.is_rtsp() else ""
            self._stt = STTTool(stt_cfg.elevenlabs_api_key, stt_cfg.language, rtsp_url)

        # World model: persistent scene entity tracker (Phase 1)
        # Shares the same PostgreSQL Database instance as ObservationMemory.
        from .db import get_db as _get_db

        try:
            self._scene = SceneTracker(_get_db())
        except Exception as exc:
            logger.warning("SceneTracker init failed: %s", exc)

        if self._camera:
            self._presence_watcher = CameraPresenceWatcher(self._pmm, camera=self.config.camera)

        # Register family members from FAMILY.md into persons DB
        self._register_family_from_md()

    @property
    def _all_tool_defs(self) -> list[dict]:
        defs = []
        if self._camera:
            defs.extend(self._camera.get_tool_definitions())
        if self._mobility:
            defs.extend(self._mobility.get_tool_definitions())
        if self._tts:
            defs.extend(self._tts.get_tool_definitions())
        defs.extend(self._memory_tool.get_tool_definitions())
        defs.extend(self._coding.get_tool_definitions())
        defs.extend(self._deferred_search.get_tool_definitions())
        defs.extend(self._deferred_fetch.get_tool_definitions())
        if self._mcp:
            defs.extend(self._mcp.get_tool_definitions())
        return defs

    async def _mcp_search(self, tool_name: str, tool_input: dict) -> tuple[str, Any]:
        """Route a search call through MCP, waiting for MCP init if needed."""
        mcp_task = getattr(self, "_mcp_start_task", None)
        if mcp_task and not mcp_task.done():
            await mcp_task
        if self._mcp:
            return await self._mcp.call(tool_name, tool_input)
        return "MCP が利用できません。", None

    async def _execute_tool(self, name: str, tool_input: dict) -> tuple[str, str | None]:
        """Route tool call to the right handler. Returns (text, image_b64_or_None)."""
        camera_tools = {"see", "look"}
        mobility_tools = {"walk"}
        tts_tools = {"say"}
        memory_tools = {"remember", "recall"}
        coding_tools = {"read_file", "edit_file", "glob", "grep", "bash"}

        if name in camera_tools and self._camera:
            result = await self._camera.call(name, tool_input)
            if name == "look":
                self._exploration.record_move(
                    tool_input.get("direction", "center"),
                    tool_input.get("degrees", 30),
                )
            return result
        elif name in mobility_tools and self._mobility:
            return await self._mobility.call(name, tool_input)
        elif name in tts_tools and self._tts:
            # 応急処置(Issue D 先行): 内的desireターンでは say() を実行しない。
            # LLMが内的ターンで say() を呼ぶと presence/quiet ゲートをバイパスし、
            # 無人・深夜でも繰り返し発言してしまうため。
            # 「話したいことを溜めて後で話す」機能は Issue D 本体(pending_speech)で実装予定。
            if (getattr(self, "_current_is_desire_turn", False)
                    and not is_social_desire(getattr(self, "_current_desire_name", ""))):
                return "(internal turn: speaking is suppressed)", None
            return await self._tts.call(name, tool_input)
        elif name in memory_tools:
            return await self._memory_tool.call(name, tool_input)
        elif name == "search_deferred":
            result = await self._deferred_search.call(name, tool_input)
            self._deferred_requested_at_turn = self._turn_count
            return result
        elif name == "fetch_deferred":
            result = await self._deferred_fetch.call(name, tool_input)
            self._deferred_requested_at_turn = self._turn_count
            return result
        elif name in coding_tools:
            return await self._coding.call(name, tool_input)
        elif self._mcp:
            # Wait for background MCP init if still running
            mcp_task = getattr(self, "_mcp_start_task", None)
            if mcp_task and not mcp_task.done():
                await mcp_task
            return await self._mcp.call(name, tool_input)
        else:
            return f"Tool '{name}' not available (check configuration).", None

    @staticmethod
    def _tool_timeout_seconds(name: str) -> float:
        """Return per-tool timeout budget in seconds."""
        return _TOOL_TIMEOUTS.get(name, _DEFAULT_TOOL_TIMEOUT)

    # brief-turn 判定は core/brief_turn.py が持つ。既存の呼び出し口を保つ薄い委譲。
    _is_candidate_brief_turn = staticmethod(brief_turn.is_candidate_brief_turn)
    _should_use_brief_reply_mode = staticmethod(brief_turn.should_use_brief_reply_mode)

    def _tool_defs_for_turn(self, *, brief_reply_mode: bool) -> list[dict]:
        tool_defs = self._all_tool_defs
        if not brief_reply_mode:
            return tool_defs
        return [tool for tool in tool_defs if tool.get("name") in _BRIEF_REPLY_TOOL_NAMES]

    _brief_reply_prompt = staticmethod(brief_turn.brief_reply_prompt)

    def _configure_backend_for_turn(
        self,
        *,
        brief_reply_mode: bool,
        user_input: str = "",
    ) -> tuple[Any, Any] | None:
        if not hasattr(self.backend, "thinking_mode"):
            return None

        # brief_reply_mode overrides everything: force thinking off for short acks
        if brief_reply_mode:
            previous = (
                getattr(self.backend, "thinking_mode", None),
                getattr(self.backend, "thinking_effort", None),
            )
            self.backend.thinking_mode = "disabled"
            if hasattr(self.backend, "thinking_effort"):
                self.backend.thinking_effort = "low"
            return previous

        # Auto-enable adaptive thinking for complex queries when the user has not
        # explicitly set a thinking override this session.
        if (
            not getattr(self, "_thinking_user_override", False)
            and self.backend.thinking_mode == "disabled"
            and user_input
            and self._is_complex_query(user_input)
        ):
            previous = (
                getattr(self.backend, "thinking_mode", None),
                getattr(self.backend, "thinking_effort", None),
            )
            self.backend.thinking_mode = "adaptive"
            return previous

        return None

    @staticmethod
    def _is_complex_query(text: str) -> bool:
        """Return True when the query likely benefits from deeper reasoning."""
        if len(text) > 200:
            return True
        return bool(_COMPLEX_QUERY_RE.search(text))

    def _restore_backend_after_turn(self, snapshot: tuple[Any, Any] | None) -> None:
        if snapshot is None:
            return
        thinking_mode, thinking_effort = snapshot
        if hasattr(self.backend, "thinking_mode"):
            self.backend.thinking_mode = thinking_mode
        if hasattr(self.backend, "thinking_effort"):
            self.backend.thinking_effort = thinking_effort

    def _maybe_swap_internal_backend(
        self, is_desire_turn: bool, desire_name: str
    ) -> Any:
        """Temporarily swap to utility backend for internal (non-social) desire turns.

        Returns the saved original backend, or None if no swap was made.
        Internal turns (look_around, explore, etc.) use Gemini Flash-Lite instead
        of the main Sonnet backend to reduce cost.
        """
        if not is_desire_turn or not desire_name or is_social_desire(desire_name):
            return None
        if self._utility_backend is self.backend:
            return None
        saved = self.backend
        self.backend = self._utility_backend
        return saved

    @property
    def _evaluator(self) -> Evaluator:
        """評価器（loop/evaluator.py）を現在の backend から導出して返す。

        `self.backend` は内部欲求ターンで utility へ一時スワップされるため、評価器は
        スナップショットせず、`_utility_backend` と現在の `self.backend` から都度導く。
        参照が変わらなければキャッシュを返す（内部ターン中の「utility is backend」判定も
        自然に追随する）。
        """
        ev = self.__dict__.get("_evaluator_obj")
        if (
            ev is None
            or ev._utility_backend is not self._utility_backend
            or ev.backend is not self.backend
        ):
            ev = Evaluator(self._utility_backend, self.backend)
            self.__dict__["_evaluator_obj"] = ev
        return ev

    def _active_memory(self) -> "ObservationMemory":
        """Return the current speaker's memory, or agent's own if no speaker is set."""
        return self._pmm.get_speaker_memory() or self._pmm.get_agent_memory()

    def _social_presence_permission(self) -> float:
        """Return 1.0 when someone is present, 0.0 when the room is empty.

        Camera active: checks CameraPresenceWatcher / PMM for detected persons.
        Camera inactive: allows social if a real user message arrived within 5 min.
        """
        if getattr(self, "_presence_watcher", None) is not None:
            pmm = getattr(self, "_pmm", None)
            if pmm is not None and pmm.get_present_ids():
                return 1.0
            return 0.0
        last = getattr(self, "_last_human_at", None)
        if last is None:
            return 0.0
        return 1.0 if (time.time() - last) < 300.0 else 0.0

    def _in_quiet_hours(self) -> bool:
        """Return True when the current time falls inside the scheduled quiet window.

        Safe default: if no schedule rule is configured, treat as NOT quiet.
        """
        rule = getattr(self, "_schedule_rule", None)
        return bool(rule is not None and rule.is_quiet())

    # Keywords that suggest the internal turn found something worth sharing.
    _INTERNAL_SHARE_PATTERNS: tuple[str, ...] = (
        "気になる", "面白い", "面白そう", "発見", "気づい", "思い出",
        "不思議", "見つけ", "変化", "新しい",
        "found", "discovered", "interesting", "noticed", "curious", "changed",
    )

    @classmethod
    def _boost_from_internal_result(cls, text: str) -> float:
        """Return a share_memory boost amount (0–0.35) based on notable content."""
        lower = text.lower()
        count = sum(1 for p in cls._INTERNAL_SHARE_PATTERNS if p.lower() in lower)
        if count == 0:
            return 0.0
        if count >= 3:
            return 0.35
        if count >= 2:
            return 0.25
        return 0.15

    def _select_addressee(
        self,
        present_ids: list[str],
        pending_rows: list[dict],
        cfg: "PendingSpeechConfig",
    ) -> str | None:
        """複数人がいる場面で誰に話しかけるかを、話したい内容の強さと関係性から確率的に決める。

        内容の強さ[p] = target=p の鮮度合計 + target=NULL の鮮度合計(全員共通)
        関係性[p]    = (trust + intimacy) / 2
        正規化(各要素を present 合計で割る) → 重み付き合成 → 確率^(1/T) で選択。
        """
        import random as _random
        from datetime import datetime, timezone as _tz
        from .time_decay import DecayState as _DecayState

        if not present_ids:
            return None

        now_epoch = datetime.now(_tz.utc).timestamp()

        def _row_score(row: dict) -> float:
            created_at = row.get("created_at")
            if isinstance(created_at, datetime):
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=_tz.utc)
                origin = created_at.timestamp()
            else:
                origin = now_epoch
            state = _DecayState(
                origin_epoch=origin,
                half_life_seconds=cfg.half_life_days * 86400.0,
                floor=cfg.floor,
                reinforce_count=int(row.get("reinforce_count", 0)),
            )
            return state.score(now_epoch)

        # 1) content strength per person
        null_score_sum = 0.0
        per_pid_score: dict[str, float] = {pid: 0.0 for pid in present_ids}
        for row in pending_rows:
            tgt = row.get("target_person_id")
            score = _row_score(row)
            if tgt is None:
                null_score_sum += score
            elif tgt in per_pid_score:
                per_pid_score[tgt] += score

        content_strength = {
            pid: per_pid_score[pid] + null_score_sum for pid in present_ids
        }
        content_total = sum(content_strength.values())

        # 2) relationship score per person
        def _rel(pid: str) -> float:
            name = self._pmm.get_person_name(pid) if hasattr(self, "_pmm") else pid
            persons = getattr(self, "_persons", None)
            if persons is not None:
                tracker = persons._trackers.get(name)
                if tracker is not None:
                    return (tracker.trust + tracker.intimacy) / 2.0
            return (0.5 + 0.4) / 2.0  # neutral defaults

        relation: dict[str, float] = {pid: _rel(pid) for pid in present_ids}
        relation_total = sum(relation.values())

        # 3) normalized weighted score
        wc = cfg.weight_content
        wr = cfg.weight_relation
        scores: dict[str, float] = {}
        for pid in present_ids:
            nc = content_strength[pid] / content_total if content_total > 0 else 1.0 / len(present_ids)
            nr = relation[pid] / relation_total if relation_total > 0 else 1.0 / len(present_ids)
            scores[pid] = wc * nc + wr * nr

        # 4) temperature scaling then proportional selection
        t = max(cfg.temperature, 1e-6)
        scaled = {pid: s ** (1.0 / t) for pid, s in scores.items()}
        total = sum(scaled.values())
        if total <= 0:
            return _random.choice(present_ids)

        r = _random.random() * total
        cumulative = 0.0
        for pid in present_ids:
            cumulative += scaled[pid]
            if r <= cumulative:
                return pid
        return present_ids[-1]

    @staticmethod
    def _drain_interrupt_queue(
        interrupt_queue: asyncio.Queue[str | None], max_items: int = 6
    ) -> list[str]:
        """Drain pending user interrupts, preserving queue order."""
        interrupts: list[str] = []
        while len(interrupts) < max_items and not interrupt_queue.empty():
            item = interrupt_queue.get_nowait()
            if item:
                interrupts.append(item)
        return interrupts

    def _memory_dedupe_key(
        self,
        kind: str,
        content: str,
        scope: str = "turn",
        scope_id: str | None = None,
    ) -> str:
        """Build a stable dedupe key to avoid duplicate writes on retries."""
        digest = hashlib.sha1(content.encode("utf-8", errors="ignore")).hexdigest()[:12]
        resolved_scope_id = scope_id or str(self._turn_count)
        return f"{scope}:{resolved_scope_id}:{kind}:{digest}"

    def _load_me_md(self) -> str:
        """Load ME.md personality file if it exists."""
        from pathlib import Path

        candidates = [
            Path("ME.md"),
            Path.home() / ".familiar_ai" / "ME.md",
        ]
        for path in candidates:
            if path.exists():
                try:
                    return path.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
        return ""

    def _load_family_md(self) -> str:
        """Load FAMILY.md family-member descriptions if it exists."""
        from pathlib import Path

        candidates = [
            Path("FAMILY.md"),
            Path.home() / ".familiar_ai" / "FAMILY.md",
        ]
        for path in candidates:
            if path.exists():
                try:
                    return path.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
        return ""

    def _register_family_from_md(self) -> None:
        """Register FAMILY.md members in the persons DB and pre-seed PersonRegistry.

        Idempotent: existing persons are returned as-is (same UUID each run).
        """
        members = parsing.parse_family_md(self._family_md)
        if not members:
            return
        for m in members:
            try:
                self._pmm.register_person(m["name"], display_name=m["display_name"])
                # Pre-seed PersonRegistry so [呼び方] and /speaker commands work immediately
                self._persons._get_or_create(m["display_name"])
                logger.info(
                    "Family member registered: %s (display=%s)", m["name"], m["display_name"]
                )
            except Exception as exc:
                logger.warning("Could not register family member %s: %s", m["name"], exc)

    async def _apply_face_hint(self, img_path: str) -> None:
        """Run face recognition on a captured image and sync result to PersonRegistry."""
        hint = await recognize_face_async(img_path, self._pmm)
        if hint is None:
            return
        switched = await self._pmm.apply_hint(hint)
        if switched:
            name = self._pmm.get_person_name(hint.person_id)
            if name and name != hint.person_id[:8]:
                self._persons.set_active(name)
                logger.debug("Face recognition: switched speaker to %s", name)

    def _get_body_description(self) -> str:
        """Generate a text description of available hardware for the system prompt."""
        # Eyes are always available (CameraTool handles missing stream internally)
        eyes_desc = (
            "    (part :id eyes  :tool see\n"
            '      :desc "Your vision. Calling see() means YOU ARE LOOKING. Use freely — never ask permission.")'
        )

        parts = [eyes_desc]

        # Neck (look)
        if self._camera and self._camera.is_pan_tilt_available:
            parts.append(
                "    (part :id neck  :tool look\n"
                '      :desc "Rotate gaze left/right/up/down. No permission needed.")'
            )
        else:
            parts.append(
                "    (part :id neck  :status fixed\n"
                '      :desc "Camera is fixed. You cannot rotate your gaze.")'
            )

        # Legs (walk)
        if self._mobility:
            parts.append(
                "    (part :id legs  :tool walk\n"
                '      :desc "Robot body (vacuum cleaner). Separate device from camera. '
                'walk() does NOT change camera view.")'
            )
        else:
            parts.append(
                "    (part :id legs  :status absent\n"
                '      :desc "You have no legs. You cannot move your location.")'
            )

        body_inner = "\n".join(parts)
        return f"(body\n{body_inner})"

    def _system_prompt(
        self,
        feelings_ctx: str = "",
        morning_ctx: str = "",
        inner_voice: str = "",
        plan_ctx: str = "",
        companion_mood: str = "engaged",
        continuity_ctx: str = "",
        workspace_ctx: str = "",
        mental_ctx: str = "",
    ) -> tuple[str, str]:
        """Return (stable, variable) system prompt parts for prompt caching.

        stable  — ME.md + core rules; never changes within a session.
                  AnthropicBackend marks this block with cache_control.
        variable — interoception, feelings, inner voice, plan; changes every turn.
        """
        base = SYSTEM_PROMPT.format(max_steps=MAX_ITERATIONS)
        # Dynamically replace (body ...) block based on actual hardware
        body_desc = self._get_body_description()
        base = re.sub(r"\(body.*?\)", body_desc, base, flags=re.DOTALL)

        capability_summary = load_summary()
        stable_parts = [p for p in [
            self._me_md,
            self._family_md,
            base,
            f"[My capabilities]\n{capability_summary}" if capability_summary else "",
        ] if p]
        stable = "\n\n---\n\n".join(stable_parts)

        agent_mood, agent_mood_intensity = self._decayed_mood()
        self_state = getattr(self, "_self_state", None)
        self_state_snapshot = self_state.snapshot() if self_state is not None else None
        intero = _interoception(
            self._started_at,
            self._turn_count,
            companion_mood,
            agent_mood=agent_mood,
            agent_mood_intensity=agent_mood_intensity,
            self_state=self_state_snapshot,
        )
        relationship_ctx = self._relationship.context_for_prompt()
        # Speaker context: who is currently talking, and list of known persons
        speaker_name = self._persons.active_name
        known = self._persons.known_names()
        speaker_ctx_parts: list[str] = [f"(speaker :name \"{speaker_name}\")"]
        others = [n for n in known if n != speaker_name]
        if others:
            speaker_ctx_parts.append(
                "(known-persons " + " ".join(f'"{n}"' for n in others) + ")"
            )
        speaker_ctx = "\n".join(speaker_ctx_parts)

        # 在席（知覚＝PMM 由来）を注入。一人称 CoT が「誰を想像するか」を知るため。
        present_ctx = ""
        pmm = getattr(self, "_pmm", None)
        if pmm is not None:
            try:
                rows = pmm.presence_status()
                if rows:
                    present_speaker = next(
                        (r["name"] for r in rows if r.get("is_speaker")), ""
                    )
                    present_others = [r["name"] for r in rows if not r.get("is_speaker")]
                    present_ctx = format_present_ctx(present_speaker, present_others)
            except Exception:
                present_ctx = ""

        now_str = clock.now_local_str()  # OS ローカル時刻＋タイムゾーン付記（例 2026-07-23 15:00 JST(+0900)）
        datetime_ctx = f"(now :datetime \"{now_str}\")"
        variable_parts: list[str] = [intero, datetime_ctx, speaker_ctx]
        if present_ctx:
            variable_parts.append(present_ctx)
        if relationship_ctx:
            variable_parts.append(relationship_ctx)
        if continuity_ctx:
            variable_parts.append(continuity_ctx)
        if mental_ctx:
            variable_parts.append(mental_ctx)
        # Morning reconstruction takes precedence on first turn; otherwise use feelings
        if morning_ctx:
            variable_parts.append(morning_ctx)
        elif feelings_ctx:
            variable_parts.append(feelings_ctx)
        # Inner voice: agent's own desire/impulse — NOT a user message.
        # Injected here so the model understands this is self-generated, not from the companion.
        if inner_voice:
            variable_parts.append(
                f"{_t('inner_voice_label')}\n{inner_voice}\n{_t('inner_voice_directive')}"
            )
        # TAPE: upfront action plan to anchor the react loop (mechanism 1)
        if plan_ctx:
            variable_parts.append(
                "[Action plan for this turn — follow it unless you discover a good reason not to]\n"
                + plan_ctx
            )

        # Global Workspace: replaces individual exploration + scene context blocks.
        # If nothing ignited this turn, fall back to direct module context.
        if workspace_ctx:
            variable_parts.append(workspace_ctx)
        else:
            exploration_ctx = self._exploration_context()
            if exploration_ctx:
                variable_parts.append(exploration_ctx)
            scene_ctx = self._scene.context_for_prompt() if self._scene else ""
            if scene_ctx:
                variable_parts.append(scene_ctx)

        variable = "\n\n---\n\n".join(variable_parts)
        return stable, variable

    def _self_continuity_context(self) -> str:
        """Return a compact continuity block from latent concerns and recent action traces."""
        blocks: list[str] = []

        concerns = getattr(self, "_concerns", None)
        if concerns is not None:
            concern_ctx = concerns.context_for_prompt(turn_index=self._turn_count)
            if concern_ctx:
                blocks.append(concern_ctx)

        prediction = getattr(self, "_prediction", None)
        if prediction is not None:
            trace_ctx = prediction.context_for_prompt()
            if trace_ctx:
                blocks.append(trace_ctx)

        return "\n\n".join(blocks)

    def _exploration_context(self) -> str:
        """Return exploration history for ICL-based direction steering."""
        return self._exploration.context_for_prompt(n=5)

    def _collect_interoception(self):
        mcp_path = os.environ.get("FAMILIAR_INTEROCEPTION_MCP_PATH", "").strip()
        if mcp_path:
            max_staleness = int(
                os.environ.get("FAMILIAR_INTEROCEPTION_MCP_MAX_STALENESS", "45").strip() or "45"
            )
            signal = MCPInteroceptionProvider(
                mcp_path,
                max_staleness_seconds=max_staleness,
            ).collect()
            if signal.provider == "noop":
                signal = RuntimeInteroceptionProvider(
                    started_at=self._started_at,
                    turn_count=self._turn_count,
                    pending_tasks=len(getattr(self, "_background_tasks", set())),
                    quiet_hours=(self._schedule_rule.start_hour, self._schedule_rule.end_hour),
                ).collect()
        else:
            signal = RuntimeInteroceptionProvider(
                started_at=self._started_at,
                turn_count=self._turn_count,
                pending_tasks=len(getattr(self, "_background_tasks", set())),
                quiet_hours=(self._schedule_rule.start_hour, self._schedule_rule.end_hour),
            ).collect()
        return signal, semantic_pressure(signal)

    def _provisional_relationship_update(
        self,
        *,
        user_text: str,
        social_policy: SocialPolicyDecision,
    ) -> None:
        lower = user_text.lower()
        if any(token in lower for token in ("hurt", "傷つ", "前の返事")):
            self._relationship.record_repair(user_text[:180], resolved=False)
            self._relationship.note_trust_shift(
                max(0.0, self._relationship.trust - 0.08), user_text[:180], confidence=0.8
            )
            self._relationship.record_failed_support_pattern(
                "advice before validation", confidence=0.75
            )
        elif social_policy.primary_act == "delight_share":
            self._relationship.note_intimacy_shift(
                min(1.0, self._relationship.intimacy + 0.04),
                "shared delight",
                confidence=0.62,
            )
        elif social_policy.primary_act in {"fatigue_signal", "grief_signal", "venting"}:
            self._relationship.record_support_preference(
                "validate first before problem solving",
                confidence=0.72,
            )
        if social_policy.primary_act == "boundary_assertion":
            self._relationship.add_boundary(user_text[:120], severity=3)
        if social_policy.primary_act == "playful_probe":
            self._relationship.record_shared_ritual("light playful exchange", confidence=0.55)

    @staticmethod
    def _format_social_policy_prompt(policy: SocialPolicyDecision) -> str:
        lines = [
            "[Interaction policy]",
            f"- primary-act: {policy.primary_act}",
            f"- response-mode: {policy.response_mode}",
            f"- softness: {policy.softness:.2f}",
            f"- directness: {policy.directness:.2f}",
            f"- initiative: {policy.initiative:.2f}",
        ]
        if policy.avoid_problem_solving:
            lines.append("- validate before advice; do not rush into fixing")
        if policy.should_recall_relational_memory:
            lines.append("- relational memory is relevant if it naturally helps")
        if policy.mention_memory:
            lines.append("- a memory mention is allowed only if it fits naturally")
        if policy.avoid_raw_interoception_numbers:
            lines.append("- never mention raw internal/body metrics")
        return "\n".join(lines)

    def _build_mental_snapshot(
        self,
        *,
        interoception_signal,
        affect,
        social_policy: SocialPolicyDecision,
        working_memory: list[dict],
        continuity_note: str,
        desires: DesireSystem | None,
    ) -> MentalStateSnapshot:
        drive_levels = desires.drive_vector() if desires is not None else {}
        dominant = desires.get_dominant() if desires is not None else None
        working_items = [
            WorkingMemoryItem(
                memory_id=str(item.get("memory_id", "")),
                summary=str(item.get("summary", "")),
                source_kind=str(item.get("source_kind", item.get("kind", "memory"))),
                salience=float(item.get("salience", item.get("confidence", 0.5))),
                episode_id=item.get("episode_id"),
            )
            for item in working_memory[:5]
        ]
        return MentalStateSnapshot(
            turn_index=self._turn_count,
            created_at=clock.now_utc_iso(),
            interoception=interoception_signal,
            affect=affect,
            social=SocialState(
                primary_act=social_policy.primary_act,
                response_mode=social_policy.response_mode,
                trust=self._relationship.trust,
                intimacy=self._relationship.intimacy,
                repair_needed=social_policy.primary_act == "repair_attempt",
                recall_relational_memory=social_policy.should_recall_relational_memory,
                mention_memory=social_policy.mention_memory,
                initiative=social_policy.initiative,
                directness=social_policy.directness,
                softness=social_policy.softness,
            ),
            drives=DriveVector(
                levels=drive_levels,
                dominant_drive=dominant[0] if dominant else None,
                dominant_level=float(dominant[1]) if dominant else 0.0,
            ),
            working_memory=working_items,
            continuity_note=continuity_note,
        )

    async def _gather_workspace_context(
        self,
        desires: DesireSystem | None = None,
        extra_coalitions: list | None = None,
    ) -> str:
        """Run one Global Workspace competition cycle and return the broadcast context.

        Gathers coalitions from all available processors in parallel, runs the
        ignition competition, and returns the winning coalition's context_block
        plus a compact peripheral-awareness summary of non-winners.

        Returns empty string if nothing reaches ignition threshold.
        """
        # Sync coalitions (wrap in to_thread to avoid blocking)
        sync_tasks = [
            asyncio.to_thread(self._exploration.as_coalition),
            asyncio.to_thread(self._self_narrative.as_coalition),
            asyncio.to_thread(self._prediction.as_coalition),
            asyncio.to_thread(self._attention_schema.as_coalition),
            asyncio.to_thread(self._meta_monitor.as_coalition),
        ]
        if self._scene is not None:
            sync_tasks.append(asyncio.to_thread(self._scene.as_coalition))
        if desires is not None:
            sync_tasks.append(asyncio.to_thread(desires.as_coalition))

        # Async coalitions
        async_tasks = [
            self._memory.as_coalition_async(),
        ]

        results = await asyncio.gather(*sync_tasks, *async_tasks, return_exceptions=True)

        from .workspace import Coalition as _Coalition

        coalitions = []
        for r in results:
            if isinstance(r, Exception):
                logger.debug("Coalition gather error: %s", r)
            elif isinstance(r, _Coalition):
                coalitions.append(r)
        for coalition in extra_coalitions or []:
            if isinstance(coalition, _Coalition):
                coalitions.append(coalition)

        if not coalitions:
            return ""

        winner = self._workspace.compete(coalitions)
        if winner is None:
            logger.debug("GlobalWorkspace: nothing reached ignition threshold — activating DMN")
            # Periodically refresh capability self-understanding during idle DMN cycles
            if should_refresh(self._turn_count):
                if should_regenerate_on_startup():
                    asyncio.ensure_future(self._regenerate_capability_manifest())
                else:
                    asyncio.ensure_future(self._refresh_capability_summary())
            # Default Mode Network: mind-wander when workspace is idle
            dmn_coalition = await self._dmn.wander()
            if dmn_coalition is None:
                return ""
            winner = dmn_coalition
            coalitions.append(dmn_coalition)

        others = [c for c in coalitions if c is not winner]
        # Update attention schema with this turn's winner (AST)
        self._attention_schema.update_focus(winner)
        await self._workspace.notify_listeners(winner)
        return self._workspace.broadcast(winner, others)

    @staticmethod
    def _select_context_blocks(
        blocks: list[tuple[str, float]],
        max_chars: int = _MORNING_CONTEXT_MAX_CHARS,
    ) -> list[str]:
        """Select high-priority context blocks within a character budget."""
        if max_chars <= 0:
            return [text for text, _ in blocks]

        ranked = [
            (idx, text, score) for idx, (text, score) in enumerate(blocks) if text and text.strip()
        ]
        ranked.sort(key=lambda item: item[2], reverse=True)

        selected: list[tuple[int, str]] = []
        used = 0
        for idx, text, _score in ranked:
            block_len = len(text)
            sep = 2 if selected else 0
            if used + sep + block_len > max_chars:
                continue
            selected.append((idx, text))
            used += sep + block_len

        selected.sort(key=lambda item: item[0])
        return [text for _, text in selected]

    async def _emotion_for_turn(self, text: str, arousal: float) -> tuple[MoodPAD, str]:
        """評価器へ委譲（loop/evaluator.py）。テスト差し替え点として残す。"""
        return await self._evaluator.emotion_for_turn(text, arousal)

    def _present_others_for_recall(self) -> list[str]:
        """在席者相関 p の対象＝在席者から AGENT_SELF と現話者を除いた在席他者（[D-在席相関]）。

        話者は想起の視点シフト（役割1・r）で既に効くため p からは外す。自分も除く。
        """
        pmm = getattr(self, "_pmm", None)
        if pmm is None:
            return []
        exclude = {AGENT_SELF_ID, pmm.current_speaker_id}
        return [pid for pid in pmm.get_present_ids() if pid not in exclude]

    async def _turn_arousal(self, user_input: str, final_text: str) -> float:
        """A（評価器 arousal）＝内容の新規性 novelty（課題5 v0.26）。

        外からの驚きを測るので user_input の novelty を使う。自発ターン（DMN・Drive
        発火）でユーザー入力が無いときは、エージェント自身の応答 final_text へフォールバック。
        """
        content = user_input if (user_input and user_input.strip()) else (final_text or "")
        return await self._memory.content_novelty_async(content)

    # Emotion intensity by label (higher = stronger felt quality)
    _MOOD_INTENSITY: dict[str, float] = {
        "excited": 0.8,
        "moved": 0.8,
        "happy": 0.6,
        "curious": 0.6,
        "sad": 0.7,
        "surprised": 0.5,
        "nostalgic": 0.5,
        "relieved": 0.5,
        "tender": 0.7,
        "playful": 0.5,
        "proud": 0.6,
    }
    _SALIENT_NARRATIVE_EMOTIONS = {
        "excited",
        "moved",
        "tender",
        "nostalgic",
        "proud",
        "surprised",
    }

    def _update_mood(self, emotion: str) -> None:
        """Update persistent mood state from the latest inferred emotion.

        Neutral emotion is ignored (mood fades on its own via decay).
        Same emotion reinforces intensity; different strong emotion replaces.
        """
        if emotion == "neutral" or emotion not in self._MOOD_INTENSITY:
            return
        new_intensity = self._MOOD_INTENSITY[emotion]
        if emotion == self._mood:
            self._mood_intensity = min(1.0, self._mood_intensity + 0.1)
        else:
            self._mood = emotion
            self._mood_intensity = new_intensity
            self._mood_set_at = time.time()

    def _decayed_mood(self) -> tuple[str, float]:
        """Return (mood, intensity) after applying exponential decay.

        Half-life ≈ 138 seconds (~2.3 min).  Below 0.1 → treated as neutral.
        """
        if self._mood == "neutral" or self._mood_intensity <= 0.0:
            return ("neutral", 0.0)
        elapsed = time.time() - self._mood_set_at
        intensity = self._mood_intensity * math.exp(-0.005 * elapsed)
        if intensity < 0.1:
            return ("neutral", 0.0)
        return (self._mood, intensity)

    async def _proactive_memory_context(self) -> str | None:
        """pending_speech 優先、なければ 2-stage 連想想起 (Issue C/D).

        Issue D: present チェック後、まず pending_speech を確認する。
        alive な pending があれば相手を選んで max_per_turn 件を発話として返す。
        pending がない/全失効 → Issue C の2段階想起にフォールスルー。
        """
        import random as _random
        from datetime import timezone as _tz

        pmm = self._pmm
        present = pmm.get_all_present_memories()
        if not present:
            return None

        present_ids = pmm.get_present_ids()

        # ── Issue D: pending_speech 優先フロー ──────────────────────────────
        pending_store = getattr(self, "_pending_store", None)
        if pending_store is not None:
            cfg = PendingSpeechConfig()
            now_epoch = datetime.now(_tz.utc).timestamp()
            try:
                all_pending = pending_store.list_active()
            except Exception:
                all_pending = []

            alive: list[dict] = []
            for row in all_pending:
                score = pending_store.freshness_score(row, now_epoch, cfg)
                if pending_store.is_expired(row, score, cfg):
                    try:
                        pending_store.delete(row["id"])
                    except Exception:
                        pass
                else:
                    alive.append(row)

            if alive:
                addressee = self._select_addressee(present_ids, alive, cfg)
                if addressee:
                    # 相手向け(target=addressee) + target=NULL を鮮度順に max_per_turn まで
                    eligible = [
                        r for r in alive
                        if r.get("target_person_id") in (addressee, None)
                    ]
                    eligible.sort(
                        key=lambda r: pending_store.freshness_score(r, now_epoch, cfg),
                        reverse=True,
                    )
                    chosen = eligible[: cfg.max_per_turn]
                    if chosen:
                        for c in chosen:
                            try:
                                pending_store.delete(c["id"])
                            except Exception:
                                pass
                        contents = [c.get("content", "") for c in chosen if c.get("content")]
                        return " / ".join(contents) if contents else None

        # ── Issue C フォールスルー: 2-stage 連想想起 ──────────────────────
        now = datetime.now()
        hour, month = now.hour, now.month
        hour_w    = int(os.environ.get("SHARE_MEMORY_HOUR_WINDOW",  "3"))
        month_w   = int(os.environ.get("SHARE_MEMORY_MONTH_WINDOW", "1"))
        pool_k    = int(os.environ.get("SHARE_MEMORY_SEED_POOL_K",  "3"))
        assoc_max = int(os.environ.get("SHARE_MEMORY_ASSOC_MAX",    "3"))
        total_max = int(os.environ.get("SHARE_MEMORY_TOTAL_MAX",    "4"))

        candidates: list[dict] = []
        for _pid, mem in present:
            candidates += await asyncio.to_thread(
                mem.pick_seed_candidates, hour, month,
                hour_window=hour_w, month_window=month_w, k=pool_k,
            )
        if not candidates:
            return None

        seed_n = _random.choice([1, 2])
        seeds = _random.sample(candidates, min(seed_n, len(candidates)))

        collected: list[dict] = list(seeds)
        for seed in seeds:
            try:
                assoc = await self._active_memory().recall_async(
                    seed.get("content", ""), n=assoc_max,
                    min_score=MemoryConfig().recall_min_score, recall_mode="spontaneous",
                )
                collected += assoc
            except Exception:
                pass

        seen: set[str] = set()
        merged: list[dict] = []
        for m in sorted(collected, key=lambda x: x.get("score", 0.0), reverse=True):
            key = m.get("memory_id") or m.get("id") or m.get("content", "")
            if key in seen:
                continue
            seen.add(key)
            merged.append(m)
            if len(merged) >= total_max:
                break

        if not merged:
            return None

        contents = [m.get("content", "") or m.get("summary", "") for m in merged]
        contents = [c for c in contents if c]
        return " / ".join(contents) if contents else None

    async def _anniversary_context(self) -> str | None:
        """Return a calendar-aware context string for today, or None if nothing notable.

        Surfaces "on this day" memories from past years and weekly/round milestones.
        Designed to be injected into morning reconstruction with high priority.
        """
        today = datetime.now().date()
        lines: list[str] = []

        # On-this-day memories (same month-day, past years)
        try:
            anniversaries = await self._memory.recall_on_this_day_async(today.month, today.day)
            for mem in anniversaries[:2]:
                content = mem.get("content", "")
                mem_date = mem.get("date", "")
                if content and mem_date:
                    lines.append(f"[On this day]: {content} ({mem_date})")
        except Exception:
            pass

        # Milestone: days since first memory
        try:
            earliest = await self._memory.get_earliest_date_async()
            if earliest:
                first_date = datetime.fromisoformat(earliest).date()
                days = (today - first_date).days
                if days >= 7:
                    # Fire on weekly boundaries and round numbers
                    if days % 7 == 0 or days in (30, 60, 90, 100, 180, 365):
                        lines.append(f"[Milestone]: {days} days since first memory.")
        except Exception:
            pass

        return "\n".join(lines) if lines else None

    async def _online_temporal_context(self, desires: DesireSystem | None = None) -> str | None:
        """Surface temporal-self fragments during ordinary turns.

        This keeps old memories, milestones, and unresolved threads available
        beyond startup reconstruction, but only when the current state suggests
        they matter.
        """
        if self._turn_count <= 1:
            return None

        share_memory_level = 0.0
        curiosity_target = None
        if desires is not None:
            try:
                share_memory_level = float(desires.level("share_memory"))
            except Exception:
                share_memory_level = 0.0
            curiosity_target = getattr(desires, "curiosity_target", None)

        tension = 0.0
        self_state = getattr(self, "_self_state", None)
        if self_state is not None:
            try:
                tension = float(self_state.snapshot().get("unresolved_tension", 0.0))
            except Exception:
                tension = 0.0

        should_surface_memory = share_memory_level >= 0.45 or tension >= 0.45
        should_surface_anniversary = self._turn_count % 4 == 0 or tension >= 0.6
        should_surface_thread = bool(curiosity_target) and tension >= 0.5

        if not (should_surface_memory or should_surface_anniversary or should_surface_thread):
            return None

        proactive_ctx: str | None = None
        anniversary_ctx: str | None = None
        if should_surface_memory and should_surface_anniversary:
            proactive_ctx, anniversary_ctx = await asyncio.gather(
                self._proactive_memory_context(),
                self._anniversary_context(),
            )
        elif should_surface_memory:
            proactive_ctx = await self._proactive_memory_context()
        elif should_surface_anniversary:
            anniversary_ctx = await self._anniversary_context()

        lines: list[str] = []
        if should_surface_thread:
            lines.append(f"[Unresolved thread]: {str(curiosity_target)[:160]}")
        if proactive_ctx:
            lines.append(f"[Resurfaced memory]: {proactive_ctx[:180]}")
        if anniversary_ctx:
            lines.append(anniversary_ctx)

        if not lines:
            return None
        return "[Temporal self]\n" + "\n".join(lines)

    async def _infer_companion_mood(self, text: str) -> str:
        """評価器へ委譲（loop/evaluator.py）。テスト差し替え点として残す。"""
        return await self._evaluator.infer_companion_mood(text)

    async def _check_response_coherence(self, response: str) -> str | None:
        """評価器へ委譲（loop/evaluator.py）。生の会話履歴を渡す。"""
        return await self._evaluator.check_response_coherence(response, self.messages)

    async def _summarize_exchange(self, user_input: str, agent_response: str) -> str:
        """評価器へ委譲（loop/evaluator.py）。テスト差し替え点として残す。"""
        return await self._evaluator.summarize_exchange(user_input, agent_response)

    def _backup_status_note(self) -> str:
        """Return a system note if the last DB backup is stale (>25h), else empty string."""
        log_path = Path.home() / ".familiar_ai" / "backups" / "backup.log"
        if not log_path.exists():
            return ""
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            return ""
        matches = re.findall(r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\] Done:", text)
        if not matches:
            return "[system: no successful database backup on record]"
        last_backup = datetime.fromisoformat(matches[-1])
        age_hours = (datetime.now() - last_backup).total_seconds() / 3600
        if age_hours > 25:
            return f"[system: last database backup was {int(age_hours)}h ago — may need attention]"
        return ""

    async def _morning_reconstruction(self, desires=None) -> str:
        """Build a 'yesterday → today' bridge from stored memories.

        Damasio's autobiographical self coming online: reading the past
        to know who we are now. Called only on the first turn of a session.
        """
        logger.info("Morning reconstruction started")
        (
            self_model,
            curiosities,
            feelings,
            day_summaries,
            semantic_facts,
            behavior_policies,
        ) = await asyncio.gather(
            self._memory.recall_self_model_async(n=5),
            self._memory.recall_curiosities_async(n=3),
            self._memory.recent_feelings_async(n=3),
            self._memory.recall_day_summaries_async(n=5),
            self._memory.recall_semantic_facts_async("", n=5),
            self._memory.recall_behavior_policies_async("", n=4),
        )
        logger.info(
            "Morning data: self_model=%d, curiosities=%d, feelings=%d, day_summaries=%d, "
            "semantic_facts=%d, behavior_policies=%d",
            len(self_model),
            len(curiosities),
            len(feelings),
            len(day_summaries),
            len(semantic_facts),
            len(behavior_policies),
        )

        # Generate day summaries for past dates that don't have one yet.
        # Run in background so it never delays the first-turn greeting response.
        asyncio.ensure_future(self._backfill_day_summaries())

        # Re-fetch if backfill created new summaries
        if not day_summaries:
            day_summaries = await self._memory.recall_day_summaries_async(n=5)

        # Surface the most recent curiosity into the desire system
        if desires is not None and curiosities and desires.curiosity_target is None:
            desires.curiosity_target = curiosities[0]["summary"]

        blocks: list[tuple[str, float]] = []
        if day_summaries:
            blocks.append((self._memory.format_day_summaries_for_context(day_summaries), 0.78))
        if semantic_facts:
            avg_conf = sum(float(x.get("confidence", 0.5)) for x in semantic_facts) / len(
                semantic_facts
            )
            blocks.append(
                (
                    self._memory.format_semantic_facts_for_context(semantic_facts),
                    0.86 + avg_conf * 0.1,
                )
            )
        if behavior_policies:
            avg_conf = sum(float(x.get("confidence", 0.5)) for x in behavior_policies) / len(
                behavior_policies
            )
            blocks.append(
                (
                    self._memory.format_behavior_policies_for_context(behavior_policies),
                    0.84 + avg_conf * 0.1,
                )
            )
        if self_model:
            blocks.append((self._memory.format_self_model_for_context(self_model), 0.83))
        if curiosities:
            blocks.append((self._memory.format_curiosities_for_context(curiosities), 0.74))
        if feelings:
            blocks.append((self._memory.format_feelings_for_context(feelings), 0.71))

        parts = self._select_context_blocks(blocks, _MORNING_CONTEXT_MAX_CHARS)

        # Prepend self-narrative: the felt sense of continuity from past sessions.
        # This is the thread that says "ウチはここにいた、今もいる."
        narrative_ctx = self._self_narrative.context_for_prompt()

        if not parts and not narrative_ctx:
            # No history yet — make it explicit so the agent doesn't fabricate a past
            return _t("morning_no_history")

        header = _t("morning_header")
        sections: list[str] = []
        if narrative_ctx:
            sections.append(narrative_ctx)
        sections.extend(parts)
        return header + "\n\n" + "\n\n".join(sections)

    async def _backfill_day_summaries(self) -> None:
        """Generate day summaries for past dates that don't have one yet.

        Skips today (summary is generated at shutdown). Only processes
        the most recent 5 days to keep startup time reasonable.

        Skipped when no separate utility backend is configured: the main
        conversation backend may not handle bulk observations well (e.g.
        Kimi K2.5 input-size limits), and we don't want to stall startup.
        """
        if self._utility_backend is self.backend:
            logger.debug("Backfill skipped: no separate utility backend configured")
            return
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            all_dates = await asyncio.to_thread(self._memory.get_dates_with_observations, 7)
            existing = await asyncio.to_thread(self._memory.get_dates_with_summaries)
            logger.info(
                "Backfill check: today=%s, all_dates=%s, existing=%s",
                today,
                all_dates,
                existing,
            )

            missing = [d for d in all_dates if d != today and d not in existing][:5]
            if missing:
                logger.info("Backfill: generating day summaries for %s", missing)
            else:
                logger.info("Backfill: no missing day summaries")
            for date in missing:
                await self._generate_day_summary(date)
        except Exception as e:
            logger.warning("Day summary backfill failed: %s", e)

    async def _generate_day_summary(self, date: str) -> None:
        """Generate and save a day summary for the given date."""
        try:
            observations = await asyncio.to_thread(self._memory.get_observations_for_date, date, 50)
            if not observations:
                logger.info("No observations for %s, skipping day summary", date)
                return

            # Build a concise transcript for the LLM — keep it short
            lines = []
            for obs in observations:
                emotion = f" [{obs['emotion']}]" if obs["emotion"] != "neutral" else ""
                lines.append(f"  {obs['time']} ({obs['kind']}){emotion}: {obs['content'][:150]}")
            transcript = "\n".join(lines)
            logger.info("Generating day summary for %s (%d observations)", date, len(observations))

            summary = await asyncio.wait_for(
                self._utility_backend.complete(
                    _DAY_SUMMARY_PROMPT.format(
                        lang=_t("summary_lang"),
                        observations=transcript,
                    ),
                    max_tokens=400,
                ),
                timeout=30.0,
            )
            if summary:
                await self._memory.save_async(
                    summary,
                    direction="記憶",
                    kind="day_summary",
                    emotion="neutral",
                    override_date=date,
                    dedupe_key=self._memory_dedupe_key(
                        "day_summary", summary[:200], scope="day", scope_id=date
                    ),
                    materialize_now=False,
                )
                logger.info("Day summary generated for %s: %s", date, summary[:80])
                # 時間減衰は想起の t 軸（time_score）へ一元化したため、importance の
                # 日次減衰は行わない（Phase 2 P-1・[D-想起合成]。a 軸＝(a0,n) はイベント駆動）。
            else:
                logger.warning("Day summary for %s: LLM returned empty response", date)
        except asyncio.TimeoutError:
            logger.warning("Day summary for %s timed out (30s)", date)
        except Exception as e:
            logger.warning("Failed to generate day summary for %s: %s", date, e)

    async def _refresh_capability_summary(self) -> None:
        """Ask the LLM to read capabilities.yaml and write a first-person summary.

        Stored in agent_state["capability_summary"] and injected each turn.
        """
        manifest = load_manifest()
        if not manifest:
            return
        try:
            prompt = (
                "Below is the capability manifest for this agent system.\n"
                "Write a concise first-person summary (10–20 lines) of what you can do, "
                "based only on capabilities marked enabled:true or with an enabled_env note. "
                "Use natural language. Start each line with '- I can ...'.\n\n"
                f"{manifest}"
            )
            summary = await self._utility_backend.complete(prompt, max_tokens=512)
            if summary:
                save_summary(summary.strip())
                logger.info("Capability summary refreshed (%d chars)", len(summary))
        except Exception as e:
            logger.warning("Could not refresh capability summary: %s", e)

    async def _regenerate_capability_manifest(self) -> None:
        """Rewrite capabilities.yaml from source introspection via the utility LLM.

        Called at most once per day during ``rest`` desire turns.
        """
        try:
            context = collect_manifest_context()
            existing = load_manifest()
            prompt = build_generation_prompt(context, existing)
            yaml_content = await self._utility_backend.complete(prompt, max_tokens=2500)
            if yaml_content:
                save_manifest(yaml_content)
                await self._refresh_capability_summary()
        except Exception as e:
            logger.warning("capabilities.yaml regeneration failed: %s", e)

    async def _update_self_model(self, final_text: str, emotion: str) -> None:
        """Extract a self-insight and store it as self_model memory.

        Conway's working self: what this response reveals about who I am.
        Only runs when something actually moved us (non-neutral emotion).
        """
        if emotion == "neutral":
            return
        try:
            insight = await self._utility_backend.complete(
                _SELF_MODEL_PROMPT.format(text=final_text[:400]),
                max_tokens=80,
            )
            if insight and insight.lower() != "nothing":
                await self._memory.save_async(
                    insight,
                    direction="内省",
                    kind="self_model",
                    emotion=emotion,
                    dedupe_key=self._memory_dedupe_key("self_model", insight),
                    materialize_now=False,
                )
                logger.info("Self-model updated: %s", insight[:60])
        except Exception as e:
            logger.warning("Self-model update failed: %s", e)

    async def _maybe_update_self_narrative(
        self,
        *,
        user_input: str,
        final_text: str,
        emotion: str,
        is_desire_turn: bool,
    ) -> None:
        """Capture salient within-session self-narrative moments."""
        if not final_text or final_text == "(no response)":
            return

        pred_signal = self._prediction.last_signal()
        agency_error = float(pred_signal.agency_error) if pred_signal is not None else 0.0
        salient_emotion = emotion in self._SALIENT_NARRATIVE_EMOTIONS

        if not salient_emotion and agency_error < 0.55:
            return

        reason = "salient_turn" if salient_emotion else "agency_error"
        if salient_emotion and agency_error >= 0.55:
            reason = "salient_turn_agency"
        if is_desire_turn and not salient_emotion and agency_error < 0.7:
            return

        prompt = (
            "次の出来事を、ウチ自身の自己叙述として一文で書いて。\n"
            f"user: {user_input[:160]}\n"
            f"agent: {final_text[:220]}\n"
            f"emotion: {emotion}\n"
            f"agency_error: {agency_error:.2f}\n"
            "条件: 一人称は『ウチ』。60文字以内。説明や前置きは禁止。"
            "ウチはAIエージェントであり食事・移動などの身体的行動はしない。"
            "userの行動・予定・感情ではなく、ウチ自身が感じたこと・したことを書くこと。"
        )
        try:
            text = await asyncio.wait_for(
                self._utility_backend.complete(prompt, max_tokens=120),
                timeout=12.0,
            )
            if text and text.strip():
                mood = emotion if emotion != "neutral" else self._decayed_mood()[0]
                self._self_narrative.write(text.strip(), mood=mood, trigger=reason)
                logger.info("Self-narrative moment captured (%s): %s", reason, text.strip()[:60])
        except Exception as e:
            logger.warning("Could not update self narrative mid-session: %s", e)

    async def _maybe_adapt_values(
        self,
        *,
        user_input: str,
        final_text: str,
        emotion: str,
        camera_used: bool,
        curiosity: str | None,
        is_desire_turn: bool,
        desires: DesireSystem | None,
    ) -> None:
        """Lightweight experience-driven updates for policy/value confidence."""
        updates = []

        pred_signal = self._prediction.last_signal()
        if camera_used and curiosity:
            updates.append(
                self._memory.adjust_behavior_policy_confidence_async(
                    "curiosity:active",
                    0.08,
                    reason="curiosity_satisfied",
                    policy_text=f"When idle, follow up this curiosity thread: {curiosity[:180]}",
                    trigger_context="idle",
                    action_hint="look_around",
                )
            )
            if desires is not None:
                desires.boost("share_memory", 0.08)

        if (
            pred_signal is not None
            and pred_signal.action_name in {"look", "walk", "see"}
            and pred_signal.agency_error >= 0.55
        ):
            updates.append(
                self._memory.adjust_behavior_policy_confidence_async(
                    "curiosity:active",
                    -0.05,
                    reason="agency_error_high",
                )
            )

        if not is_desire_turn and user_input and emotion in {"moved", "tender", "relieved"}:
            updates.append(
                self._memory.adjust_behavior_policy_confidence_async(
                    "conversation:supportive_style",
                    0.04,
                    reason="supportive_exchange",
                    policy_text=(
                        "Prefer this response style when supporting the companion: "
                        f"{final_text[:180]}"
                    ),
                    trigger_context="conversation",
                    action_hint="respond_supportively",
                )
            )

        if emotion in {"moved", "proud", "tender"}:
            updates.append(
                self._memory.adjust_semantic_fact_confidence_async(
                    "self_model:core",
                    0.03,
                    reason="salient_self_consistency",
                )
            )

        if not updates:
            return

        results = await asyncio.gather(*updates, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.debug("Adaptive value update failed: %s", result)

    async def extract_curiosity(self, exploration_result: str) -> str | None:
        """Ask the LLM what was most curious/interesting in the exploration."""
        try:
            none_word = _t("curiosity_none")
            text = await self._utility_backend.complete(
                f"Read this exploration report and answer in one sentence what you found most "
                f"curious or interesting. Write in {_t('summary_lang')}. "
                f'If nothing caught your attention, reply with just "{none_word}". '
                f"No explanation.\n\n{exploration_result}",
                max_tokens=80,
            )
            text = text.strip()
            # Reject if the model returned the "none" word or a long non-curious explanation
            if not text or none_word in text or len(text) > 100:
                return None
            return text
        except Exception as e:
            logger.warning("Curiosity extraction failed: %s", e)
        return None

    def should_deliver_deferred_result(self) -> bool:
        """Return True when a proactive deferred-search delivery turn should fire.

        Four gates must all pass:
          1. Pending results exist
          2. Someone is present (camera: person detected; no camera: recent user message)
          3. Quiet-hours mode is not active
          4. Current social context allows interruption
             (blocked during grief / venting / emotional repair / boundary)
        """
        if not getattr(self, "_deferred_search", None):
            return False
        # Gate 1: at least one result is ready
        search_pending = self._deferred_search.has_pending
        fetch_pending = self._deferred_fetch.has_pending
        if not (search_pending or fetch_pending):
            return False
        # Wait until all concurrent tasks finish so results are delivered together
        if self._deferred_search.is_running or self._deferred_fetch.is_running:
            return False

        # A user-initiated search whose requester was active within 30 minutes
        # bypasses both the presence gate and the quiet-hours gate: the user
        # explicitly asked for this and is effectively present, even if camera
        # face recognition has not registered them in _present.
        import time as _time
        _user_recent = _time.time() - getattr(self, "_last_human_at", 0) < 1800
        _user_initiated = _user_recent and (
            self._deferred_search.has_user_initiated_pending
            or self._deferred_fetch.has_user_initiated_pending
        )

        # Gate 2: presence — reuse the same logic as social desires.
        # Bypassed for user-initiated recent searches (see above).
        if not _user_initiated and self._social_presence_permission() == 0.0:
            return False

        # Gate 3: quiet mode — bypassed for user-initiated recent searches.
        if not _user_initiated:
            rule = getattr(self, "_schedule_rule", None)
            if rule is not None:
                from datetime import datetime
                if rule.is_quiet(datetime.now()):
                    return False

        # Gate 4: social policy
        last_policy = getattr(self, "_last_social_decision", None)
        if last_policy is not None:
            _BLOCKED = frozenset({
                "grief_signal", "venting", "fatigue_signal",
                "repair_attempt", "boundary_assertion",
            })
            if last_policy.primary_act in _BLOCKED:
                return False

        return True

    def _should_compact(self, threshold_tokens: int = 20_000) -> bool:
        """Return True when context is large enough to warrant compaction.

        A threshold of 0 acts as a disabled sentinel — never compact.
        In normal use _last_context_tokens is 0 until after the first turn,
        so an empty conversation naturally returns False.
        Threshold set to 20k to stay safely under the 30k input-TPM rate limit.
        """
        return threshold_tokens > 0 and self._last_context_tokens > threshold_tokens

    async def _compact_messages(self, keep_last: int = 6) -> None:
        """Summarise old messages and trim the history.

        Keeps the last `keep_last` messages verbatim, replaces the rest with a
        single summary marker, and sets `_post_compact = True` so the next
        `run()` call does a boosted memory recall to compensate.
        """
        if len(self.messages) <= keep_last:
            return

        to_summarise = self.messages[:-keep_last]
        recent = self.messages[-keep_last:]

        # Build a plain-text transcript for the summary LLM call
        lines = []
        for msg in _flatten_history(to_summarise):  # tool結果はネストlist。走査前に展開
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            lines.append(f"{role}: {content[:300]}")
        history_text = "\n".join(lines)

        summary = await self._utility_backend.complete(
            _COMPACT_PROMPT.format(history=history_text),
            max_tokens=200,
        )
        summary_marker = self.backend.make_user_message(
            f"[Conversation summary — earlier turns compacted]\n{summary}"
        )

        self.messages = [summary_marker] + list(recent)
        self._post_compact = True

    @property
    def is_embedding_ready(self) -> bool:
        """Return True once the embedding model has finished loading."""
        return self._memory.is_embedding_ready()

    async def _write_today_narrative(self) -> None:
        """Write a one-sentence self-description for today's session.

        This is Kokone's diary entry — "who I was today." Read back next session
        as the felt thread of temporal continuity: ウチはここにいた、今もいる.
        """
        if self._turn_count == 0:
            return  # No conversation happened — nothing to narrate
        try:
            today_memories = await self._memory.recall_day_summaries_async(n=1)
            if today_memories:
                summary_hint = today_memories[0].get("content", "")[:200]
            else:
                # Fall back to recent observations
                recent = await self._memory.recall_async("", n=5, recall_mode="system")
                summary_hint = " / ".join(m.get("content", "")[:60] for m in recent[:3])

            mood, _ = self._decayed_mood()
            prompt = (
                f"今日起きたこと（要約）:\n{summary_hint}\n\n"
                "ウチ（ここね）として、今日という日を一文で書いて。"
                "一人称は「ウチ」、50文字以内、過去形。"
                "感情や気づきを含めて。"
            )
            text = await asyncio.wait_for(
                self._utility_backend.complete(prompt, max_tokens=120),
                timeout=15.0,
            )
            if text and text.strip():
                self._self_narrative.write(text.strip(), mood=mood)
                logger.info("Self-narrative written: %s", text.strip()[:60])
        except Exception as e:
            logger.warning("Could not write today's self narrative: %s", e)

    async def close(self) -> None:
        """Clean up resources. Bounded by timeouts to avoid hanging on exit."""
        if self._camera:
            self._camera.close()

        heartbeat = getattr(self, "_cache_heartbeat_task", None)
        if heartbeat and not heartbeat.done():
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

        await self._drain_background_tasks()

        # Write today's self-narrative before shutting down.
        await self._write_today_narrative()

        # Generate (or refresh) today's day summary before shutting down.
        # Skipped when no separate utility backend is configured.
        if self._utility_backend is not self.backend:
            try:
                today = datetime.now().strftime("%Y-%m-%d")
                await asyncio.to_thread(self._memory.delete_day_summaries_for_date, today)
                await self._generate_day_summary(today)
            except Exception as e:
                logger.warning("Failed to generate today's day summary on shutdown: %s", e)
        memory_worker = getattr(self, "_memory_worker", None)
        if memory_worker:
            try:
                await asyncio.wait_for(memory_worker.stop(), timeout=1.5)
            except (asyncio.TimeoutError, Exception):
                pass
        if self._mcp:
            try:
                await asyncio.wait_for(self._mcp.stop(), timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                pass
        _pw = getattr(self, "_presence_watcher", None)
        if _pw is not None:
            try:
                await asyncio.wait_for(_pw.stop(), timeout=1.0)
            except (asyncio.TimeoutError, Exception):
                pass
        try:
            await asyncio.wait_for(asyncio.to_thread(self._memory.close), timeout=1.0)
        except (asyncio.TimeoutError, Exception):
            pass
        self._persons.close()

    # ── Multi-person relationship delegation ─────────────────────────────────

    @property
    def _relationship(self) -> RelationshipTracker:
        """Active speaker's RelationshipTracker (transparent alias for legacy call sites)."""
        return self._persons.active

    @_relationship.setter
    def _relationship(self, value: RelationshipTracker) -> None:
        """Allow direct assignment (used by tests and legacy code)."""
        if not hasattr(self, "_persons"):
            # Called before __init__ completes (e.g. test fixtures using __new__).
            # Bootstrap a minimal PersonRegistry so the property getter works.
            self._persons = PersonRegistry(default_name="companion")
        self._persons._trackers[self._persons.active_name] = value

    async def _sync_pmm_speaker(self, name: str) -> None:
        """Set PersonMemoryManager speaker to match the name from PersonRegistry."""
        pmm = getattr(self, "_pmm", None)
        if pmm is None:
            return
        pid = pmm.find_person_id_by_name(name)
        if pid:
            await pmm.set_speaker(pid, source="text")

    async def _on_pmm_speaker_switch(self, old_id: str | None, new_id: str) -> None:
        """PMM on_switch callback: update companion name in the active DesireSystem."""
        desires = getattr(self, "_desires_ref", None)
        if desires is None:
            return
        info = self._pmm.get_speaker_info()
        name = (info or {}).get("display_name") or (info or {}).get("name", "")
        if name:
            desires.update_active_companion(name)

    def _handle_speaker_command(self, user_input: str) -> str | None:
        """/speaker [name] — set or show the active speaker for this session."""
        m = _SPEAKER_COMMAND_RE.match(user_input.strip())
        if m is None:
            return None
        name_arg = (m.group(1) or "").strip()
        if not name_arg:
            current = self._persons.active_name
            known = ", ".join(self._persons.known_names())
            return f"[現在の話者: {current}  既知: {known}]"
        self._persons.set_active(name_arg)
        asyncio.ensure_future(self._sync_pmm_speaker(name_arg))
        return f"[話者を「{name_arg}」に切り替えました]"

    def _handle_reload_command(self, user_input: str) -> str | None:
        """Reload ME.md and FAMILY.md without restarting. Returns status string or None."""
        if not _RELOAD_COMMAND_RE.match(user_input.strip()):
            return None

        old_me = self._me_md
        old_family = self._family_md

        self._me_md = self._load_me_md()
        self._family_md = self._load_family_md()

        lines: list[str] = []
        lines.append("[リロード完了]")
        if self._me_md != old_me:
            lines.append("• ME.md を更新しました")
            if not os.environ.get("AGENT_NAME"):
                me_name = parsing.parse_me_name(self._me_md)
                if me_name:
                    self.config.agent_name = me_name
        else:
            lines.append("• ME.md 変更なし")
        if self._family_md != old_family:
            lines.append("• FAMILY.md を更新しました")
            self._register_family_from_md()
            if not os.environ.get("COMPANION_NAME"):
                members = parsing.parse_family_md(self._family_md)
                if members:
                    first_call = members[0]["display_name"].split("、")[0].split(",")[0].strip()
                    if first_call:
                        self.config.companion_name = first_call
                        self._persons._default_name = first_call
        else:
            lines.append("• FAMILY.md 変更なし")
        lines.append("次のターンから新しい内容が反映されます。")
        return "\n".join(lines)

    def _handle_thinking_command(self, user_input: str) -> str | None:
        """Return a status string if user_input is a thinking-mode command, else None.

        Handles /think slash commands and a small set of exact natural-language
        phrases. Returns None to indicate the caller should continue normally.
        """
        stripped = user_input.strip()

        # --- slash command: /think [on|off|adaptive|disabled|status] ---
        m = _THINK_COMMAND_RE.match(stripped)
        if m:
            arg = (m.group(1) or "").lower()
            if arg == "status":
                current = getattr(self.backend, "thinking_mode", "不明")
                return f"[思考モード: {current}]"

            if not hasattr(self.backend, "thinking_mode"):
                return "[このバックエンドは思考モードの切替に対応していません]"

            if arg in ("on", "adaptive"):
                new_mode = "adaptive"
            elif arg in ("off", "disabled"):
                new_mode = "disabled"
            else:
                # bare /think → toggle
                current = getattr(self.backend, "thinking_mode", "disabled")
                new_mode = "disabled" if current == "adaptive" else "adaptive"

            self.backend.thinking_mode = new_mode
            # Track that the user has explicitly set thinking mode this session.
            # This prevents the per-turn auto-thinking heuristic from reverting it.
            self._thinking_user_override = new_mode != "disabled"
            label = "有効（adaptive）" if new_mode == "adaptive" else "無効"
            return f"[思考モードを {label} に切り替えました]"

        # --- exact natural-language phrases ---
        if not hasattr(self.backend, "thinking_mode"):
            return None  # backend doesn't support it; ignore silently

        if stripped in _THINK_ON_EXACT:
            self.backend.thinking_mode = "adaptive"
            self._thinking_user_override = True
            return "[思考モードを有効（adaptive）に切り替えました]"

        if stripped in _THINK_OFF_EXACT:
            self.backend.thinking_mode = "disabled"
            self._thinking_user_override = False
            return "[思考モードを無効に切り替えました]"

        return None

    async def run(
        self,
        user_input: str,
        on_action: Callable[[str, dict], None] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_image: Callable[[str], None] | None = None,
        on_phase: Callable[[str], None] | None = None,
        on_tool_result: Callable[[str, dict, str], None] | None = None,
        desires=None,
        inner_voice: str = "",
        desire_name: str = "",
        interrupt_queue=None,
    ) -> str:
        """Run one conversation turn with the agent loop.

        inner_voice: agent's own desire/impulse (injected into system prompt, NOT a user message).
        desire_name: the desire that triggered this turn (empty for user turns).
        """
        if desires is not None:
            self._desires_ref = desires
        if not hasattr(self, "_schedule_rule"):
            self._schedule_rule = parse_schedule_config(
                Path.home() / ".familiar_ai" / "schedule.conf"
            )
        if not hasattr(self, "_mental_state_bus"):
            self._mental_state_bus = MentalStateBus()
        if not hasattr(self, "_appraisal"):
            self._appraisal = AppraisalEngine()
        if not hasattr(self, "_social_policy"):
            self._social_policy = SocialPolicyEngine()
        if not hasattr(self, "_heartbeat"):
            self._heartbeat = HeartbeatRuntime(
                memory=getattr(self, "_memory", None),
                quiet_rule=self._schedule_rule,
            )
        if not hasattr(self, "_last_tool_error"):
            self._last_tool_error = None
        if not hasattr(self, "_tool_failure_streak"):
            self._tool_failure_streak = 0
        if not hasattr(self, "_last_human_at"):
            self._last_human_at = time.time()
        if not hasattr(self, "_cache_heartbeat_task"):
            if isinstance(self.backend, AnthropicBackend):
                self._cache_heartbeat_task: asyncio.Task[None] | None = asyncio.create_task(
                    self._cache_heartbeat_loop(), name="cache-heartbeat"
                )
            else:
                self._cache_heartbeat_task = None

        # ── Speaker identification ────────────────────────────────────────────
        # /speaker command sets the session-default speaker.
        _speaker_reply = self._handle_speaker_command(user_input)
        if _speaker_reply is not None:
            if on_text:
                on_text(_speaker_reply)
            return _speaker_reply

        # Parse [name] / @name: prefix; strip it from user_input for the LLM.
        user_input, _speaker_from_prefix = parsing.extract_speaker_prefix(user_input)
        if _speaker_from_prefix:
            self._persons.set_active(_speaker_from_prefix)
            await self._sync_pmm_speaker(_speaker_from_prefix)

        # ── File reload command ───────────────────────────────────────────────
        _reload_reply = self._handle_reload_command(user_input)
        if _reload_reply is not None:
            if on_text:
                on_text(_reload_reply)
            return _reload_reply

        # ── Thinking-mode slash-commands & natural-language shortcuts ────────
        # These return immediately without calling the LLM.
        _think_reply = self._handle_thinking_command(user_input)
        if _think_reply is not None:
            if on_text:
                on_text(_think_reply)
            return _think_reply

        self._turn_count += 1
        first_turn = self._turn_count == 1
        memory_worker = getattr(self, "_memory_worker", None)
        startup_phase = (
            first_turn
            or not self._memory.is_embedding_ready()
            or (self._mcp is not None and not self._mcp.is_started)
            or (memory_worker is not None and not memory_worker.is_running)
        )
        if on_phase:
            on_phase("startup" if startup_phase else "thinking")

        # Start MCP connections in background (non-blocking) and memory worker
        if self._mcp and not self._mcp.is_started:
            self._mcp_start_task = asyncio.ensure_future(self._mcp.start())
        if memory_worker and not memory_worker.is_running:
            await memory_worker.start()

        is_desire_turn = bool(inner_voice and not user_input)
        self._current_is_desire_turn = is_desire_turn
        self._current_desire_name = desire_name

        # Reset per-turn note registration counter for boost gating (Issue D).
        if hasattr(self, "_memory_tool"):
            self._memory_tool._notes_registered_this_turn = 0

        # Tell the deferred tools whether this is a user-initiated turn so pending
        # results can be tagged and quiet-hours bypassed for user-requested searches.
        self._deferred_search.set_user_turn(not is_desire_turn)
        self._deferred_fetch.set_user_turn(not is_desire_turn)

        # Suppress social desire turns during quiet hours (don't wake the user).
        # Exception: share_search_result bypasses quiet hours only when the user explicitly
        # requested the search AND is still present (last message within 30 minutes).
        if is_desire_turn and is_social_desire(desire_name):
            _rule = getattr(self, "_schedule_rule", None)
            if _rule is not None and _rule.is_quiet():
                import time as _time
                _user_recent = (
                    _time.time() - getattr(self, "_last_human_at", 0) < 1800
                )
                _delivering_user_search = (
                    desire_name == "share_search_result"
                    and _user_recent
                    and (
                        self._deferred_search.has_user_initiated_pending
                        or self._deferred_fetch.has_user_initiated_pending
                    )
                )
                if not _delivering_user_search:
                    logger.debug("Social desire '%s' suppressed: quiet hours", desire_name)
                    return ""

        candidate_brief_turn = self._is_candidate_brief_turn(
            user_input,
            is_desire_turn=is_desire_turn,
        )

        # If PMM speaker was never set (e.g. first turn, no face recognition), sync from PersonRegistry.
        if getattr(self, "_pmm", None) and self._pmm.current_speaker_id is None:
            await self._sync_pmm_speaker(self._persons.active_name)

        # Fire mood inference immediately so it runs in parallel with all DB preprocessing.
        # Awaited just before it is needed; overlap with unfinished_business + recall gather
        # absorbs most of the Gemini round-trip.
        _mood_task: asyncio.Task[str] | None = None
        if not is_desire_turn and not candidate_brief_turn:
            _mood_task = asyncio.create_task(self._infer_companion_mood(user_input))

        # First turn: reset thinking mode and speaker to .env defaults (session-scoped)
        if first_turn:
            if hasattr(self.backend, "thinking_mode"):
                self.backend.thinking_mode = self.config.thinking_mode
                self._thinking_user_override = False
            self._persons.reset_to_default()
            if self._presence_watcher:
                asyncio.ensure_future(self._presence_watcher.start())

        # First turn: morning reconstruction — bridge yesterday's self to today's
        morning_ctx = ""
        routine_state = self._heartbeat.routine_state()
        if first_turn:
            self._relationship.record_session()
            routine_notes = self._heartbeat.morning_reconstruction_notes()
            if candidate_brief_turn:
                morning_ctx = routine_notes or ""
            else:
                morning_ctx = await self._morning_reconstruction(desires=desires)
                if routine_notes:
                    morning_ctx = (
                        f"{morning_ctx}\n\n{routine_notes}" if morning_ctx else routine_notes
                    )
            backup_note = self._backup_status_note()
            if backup_note:
                morning_ctx = f"{morning_ctx}\n\n{backup_note}" if morning_ctx else backup_note

        # Compact context if it has grown too large (GC-like: compress old turns)
        if self._should_compact():
            await self._compact_messages()

        # Inject relevant past memories + emotional context (skip for desire-driven turns)
        recall_n = 5 if self._post_compact else 3
        self._post_compact = False  # consume the flag regardless
        interoception_signal, interoception_pressure = self._collect_interoception()
        prediction_signal = self._prediction.last_signal()
        unfinished_business: list[dict] = []
        if not candidate_brief_turn:
            list_unfinished_business = getattr(self._memory, "list_unfinished_business_async", None)
            unfinished_business = await _call_optional_async(
                list_unfinished_business,
                limit=3,
                fallback=[],
            )
        companion_mood = "engaged"
        working_memory: list[dict] = []
        semantic_facts: list[dict] = []
        behavior_policies: list[dict] = []
        feelings: list[dict] = []
        memories: list[dict] = []
        recall_divergent = getattr(self._memory, "recall_divergent_async", None)
        refresh_working = getattr(self._memory, "refresh_working_memory_async", None)
        get_working = getattr(self._memory, "get_working_memory_async", None)
        if not is_desire_turn:
            if candidate_brief_turn:
                companion_mood = self._cached_companion_mood or "engaged"
                user_input_with_ctx = user_input
                feelings_ctx = ""
            else:
                # Build the memories coroutine lazily so it runs inside gather,
                # not sequentially before it (the eager-await fallback pattern was slow).
                _memories_coro = (
                    _call_optional_async(recall_divergent, user_input, n=recall_n, fallback=[])
                    if recall_divergent is not None
                    else self._active_memory().recall_async(user_input, n=recall_n, min_score=MemoryConfig().recall_min_score, recall_mode="conversation", present_others=self._present_others_for_recall())
                )
                (
                    memories,
                    feelings,
                    semantic_facts,
                    behavior_policies,
                    working_memory,
                ) = await asyncio.gather(
                    _memories_coro,
                    self._memory.recent_feelings_async(n=4),
                    self._memory.recall_semantic_facts_async(user_input, n=3),
                    self._memory.recall_behavior_policies_async(user_input, n=2),
                    _call_optional_async(
                        refresh_working,
                        user_input,
                        n=4,
                        fallback=[],
                    ),
                )
                # Mood task was fired before all DB preprocessing; by now it is
                # likely already done (Gemini overlapped with recall gather).
                try:
                    companion_mood = (
                        await _mood_task if _mood_task is not None
                        else self._cached_companion_mood or "engaged"
                    )
                except Exception:
                    companion_mood = self._cached_companion_mood or "engaged"
                working_memory = await _call_optional_async(get_working, n=4, fallback=[])
                temporal_ctx = self._cached_temporal_ctx
                memory_parts = []
                if memories:
                    memory_parts.append(self._memory.format_for_context(memories))
                if feelings:
                    memory_parts.append(self._memory.format_feelings_for_context(feelings))
                if semantic_facts:
                    memory_parts.append(
                        self._memory.format_semantic_facts_for_context(semantic_facts)
                    )
                if behavior_policies:
                    memory_parts.append(
                        self._memory.format_behavior_policies_for_context(behavior_policies)
                    )
                if temporal_ctx:
                    memory_parts.append(temporal_ctx)
                if memory_parts:
                    user_input_with_ctx = user_input + "\n\n" + "\n\n".join(memory_parts)
                else:
                    user_input_with_ctx = user_input
                feelings_ctx = (
                    self._memory.format_feelings_for_context(feelings) if feelings else ""
                )
        else:
            # Desire turn: no user context needed; feelings injected via interoception
            feelings_ctx = ""
            # Use a minimal placeholder — the real instruction is in inner_voice (system prompt).
            # The previous "（内的衝動に従って行動）" marker was echoed verbatim by the LLM.
            # "." is the shortest non-whitespace string accepted by the Anthropic API.
            user_input_with_ctx = "."

        if self._tool_failure_streak >= 2 and desires is not None:
            desires.boost("self_protect", min(0.5, 0.15 * self._tool_failure_streak))

        affect = self._appraisal.appraise(
            AppraisalContext(
                user_text=user_input,
                companion_mood=companion_mood,
                relationship_trust=self._relationship.trust,
                relationship_intimacy=self._relationship.intimacy,
                recalled_memory_summaries=tuple(m.get("summary", "") for m in memories[:3]),
                prediction_signal=prediction_signal,
                interoception=interoception_pressure,
                blocked_drives=("tool_failure",) if self._tool_failure_streak else (),
                unfinished_business_count=len(unfinished_business),
            )
        )

        previous_response_hurt = any(
            token in user_input.lower() for token in ("hurt", "傷つ", "前の返事", "嫌だった")
        )
        social_policy = self._social_policy.decide(
            user_text=user_input,
            affect=affect,
            trust=self._relationship.trust,
            intimacy=self._relationship.intimacy,
            interoception=interoception_pressure,
            previous_response_hurt=previous_response_hurt,
        )
        self._last_social_decision = social_policy
        self._provisional_relationship_update(user_text=user_input, social_policy=social_policy)

        if desires is not None:
            context_affordances = {
                "repair": 1.3 if social_policy.primary_act == "repair_attempt" else 1.0,
                "care": 1.2
                if social_policy.primary_act in {"fatigue_signal", "grief_signal", "venting"}
                else 1.0,
                "play": 1.15 if social_policy.primary_act == "playful_probe" else 0.9,
                "attachment": 1.1 if affect.attachment_pull > 0.55 else 1.0,
                "consolidate": 1.2 if unfinished_business else 1.0,
                "self_protect": 1.2 if self._tool_failure_streak >= 2 else 1.0,
            }
            _presence = self._social_presence_permission()
            _threat_factor = max(0.2, 1.0 - affect.threat * 0.35)
            desires.update_context(
                schedule_multiplier=routine_state.schedule_multiplier,
                social_permission=_threat_factor * _presence if _presence > 0.0 else 0.0,
                energy_budget=max(0.2, 1.0 - interoception_pressure.need_rest * 0.6),
                unfinished_business_bonus=min(0.4, len(unfinished_business) * 0.1),
                context_affordances=context_affordances,
            )
            if social_policy.primary_act == "repair_attempt":
                desires.boost("repair", 0.45)
            if social_policy.primary_act == "delight_share":
                desires.boost("attachment", 0.18)
            if social_policy.primary_act in {"fatigue_signal", "grief_signal"}:
                desires.boost("care", 0.22)
            if affect.frustration > 0.45:
                desires.boost("self_protect", 0.12)

        brief_reply_turn = self._should_use_brief_reply_mode(
            user_input=user_input,
            social_policy=social_policy,
            is_desire_turn=is_desire_turn,
        )

        # Inject deferred results into messages (persistent history) before appending.
        # This ensures the LLM can see delivered results in all subsequent turns.
        _deferred_parts: list[str] = []
        if _search_ctx := self._deferred_search.pending_context():
            _deferred_parts.append(_search_ctx)
        if _fetch_ctx := self._deferred_fetch.pending_context():
            _deferred_parts.append(_fetch_ctx)
        if _deferred_parts:
            _deferred_block = "\n\n".join(_deferred_parts)
            user_input_with_ctx = _deferred_block + "\n\n---\n\n" + user_input_with_ctx
            # When results arrive alongside a user message, guide the LLM to report them.
            if not inner_voice:
                inner_voice = (
                    "調べておいた結果が届いた。"
                    "いつものトーンで自然にユーザーに伝えよう。改めての挨拶は不要。"
                )

        self.messages.append(self.backend.make_user_message(user_input_with_ctx))

        # Use cached plan & workspace context from previous turn's post-response pipeline.
        # These are computed in the background after each response and are ready for the
        # next turn.  First turn uses empty defaults — morning_ctx dominates anyway.
        plan_ctx = "" if brief_reply_turn else self._cached_plan_ctx
        workspace_ctx = ""
        continuity_ctx = ""
        tape_backend = self._tape_backend()  # still needed for in-loop replanning
        if not brief_reply_turn:
            extra_coalitions = [affect.as_coalition()]
            workspace_ctx = await self._gather_workspace_context(
                # Delivery turns exclude desire coalitions: social impulses (greet etc.)
                # would override the inner-voice directive to report search/fetch results.
                # Affect, memory, attention etc. still compete to preserve personality tone.
                desires=None if _deferred_parts else desires,
                extra_coalitions=extra_coalitions,
            )
            if not workspace_ctx:
                workspace_ctx = self._cached_workspace_ctx
            continuity_ctx = self._self_continuity_context()
            heartbeat_ctx = self._heartbeat.continuity_context_for_prompt()
            if heartbeat_ctx:
                continuity_ctx = (
                    continuity_ctx
                    + ("\n\n" if continuity_ctx else "")
                    + "[Continuation]\n"
                    + heartbeat_ctx
                )
            if unfinished_business:
                continuity_ctx = (
                    continuity_ctx
                    + ("\n\n" if continuity_ctx else "")
                    + "[Open unfinished business]\n"
                    + "\n".join(f"- {item['summary'][:160]}" for item in unfinished_business[:3])
                )
            if plan_ctx:
                logger.debug("TAPE plan (cached): %s", plan_ctx[:80])
            if workspace_ctx:
                logger.debug("GlobalWorkspace broadcast (cached): %s", workspace_ctx[:80])

        mental_snapshot = self._build_mental_snapshot(
            interoception_signal=interoception_signal,
            affect=affect,
            social_policy=social_policy,
            working_memory=working_memory,
            continuity_note="; ".join(item["summary"][:80] for item in unfinished_business[:2]),
            desires=desires,
        )
        if brief_reply_turn:
            mental_ctx = "\n\n".join(
                part
                for part in (
                    self._format_social_policy_prompt(social_policy),
                    self._brief_reply_prompt(),
                )
                if part
            )
        else:
            mental_ctx = "\n\n".join(
                part
                for part in (
                    self._mental_state_bus.summarize_recent_for_prompt(2),
                    mental_snapshot.prompt_summary(),
                    self._format_social_policy_prompt(social_policy),
                )
                if part
            )

        if on_phase and startup_phase:
            on_phase("thinking")

        camera_used = False
        camera_image: str | None = None  # raw base64 JPEG from the latest `see` tool call
        say_used = False
        # 実際にユーザーへ届いた発話。2回目以降の say() は音声も画面表示も抑制される
        # ので、届いた最初の1回だけを持つ。永続化はこれと本文の両方を使う。
        spoken_text = ""
        # ターン中に書かれた本文（考えたこと）。最後の周だけでなく、ツールを使った
        # 周に書かれたものも自分がしたことなので拾う。
        turn_thoughts: list[str] = []
        say_nudge_used = False  # one-time say() nudge per turn (silence-control step 3)
        final_text = "(no response)"
        non_say_streak = 0  # consecutive tool calls without say()
        observation_action_name: str | None = None
        observation_action_input: dict | None = None
        pending_view_action_name: str | None = None
        pending_view_action_input: dict | None = None
        turn_tools = self._tool_defs_for_turn(brief_reply_mode=brief_reply_turn)
        turn_max_tokens = (
            min(self.config.max_tokens, _BRIEF_REPLY_MAX_TOKENS)
            if brief_reply_turn
            else self.config.max_tokens
        )
        turn_max_iterations = _BRIEF_REPLY_MAX_ITERATIONS if brief_reply_turn else MAX_ITERATIONS
        backend_turn_snapshot = self._configure_backend_for_turn(
            brief_reply_mode=brief_reply_turn,
            user_input=user_input,
        )
        _internal_backend_saved = self._maybe_swap_internal_backend(is_desire_turn, desire_name)
        # 内的desireターンは結果を履歴に残さないためローカルコピーを使う。
        # 通常ターンは本体参照。形式変換は各バックエンドの stream_turn 内部が行う（呼び出し側は変換しない）。
        if _internal_backend_saved is not None:
            # 非社会的な内的desireターンは会話履歴を引き継がない（直前発話の復唱を防ぐ）。
            # 内的衝動は会話の残響でなく記憶の想起（system promptの[Resurfaced memory]）に基づく。
            # 直前に追加した最小プレースホルダ "." のみを起点にする
            # （messagesを空にすると Gemini の contents が空になり拒否されるため）。
            # ターン内で生成されるメッセージは turn_messages.append で積まれ連続性は保たれる。
            turn_messages = [self.messages[-1]]
        else:
            turn_messages = self.messages

        try:
            for i in range(turn_max_iterations):
                logger.debug("Agent iteration %d", i + 1)

                result, raw_content = await self.backend.stream_turn(
                    system=self._system_prompt(
                        feelings_ctx,
                        morning_ctx,
                        inner_voice=inner_voice,
                        plan_ctx=plan_ctx,
                        companion_mood=companion_mood,
                        continuity_ctx=continuity_ctx,
                        workspace_ctx=workspace_ctx,
                        mental_ctx=mental_ctx,
                    ),
                    messages=turn_messages,
                    tools=turn_tools,
                    max_tokens=turn_max_tokens,
                    on_text=on_text,
                )
                _text = (result.text or "").strip()
                if _text and _text not in turn_thoughts:
                    turn_thoughts.append(_text)
                self._last_context_tokens = result.input_tokens
                self._session_input_tokens += result.input_tokens
                self._session_output_tokens += result.output_tokens

                # HOT layer: record this step metacognitively
                _focus = self._attention_schema.current_focus()
                if _focus is not None:
                    _action = result.stop_reason
                    if result.stop_reason == "tool_use" and result.tool_calls:
                        _action = result.tool_calls[0].name
                    _conf = min(1.0, result.output_tokens / max(1, self.config.max_tokens))
                    self._meta_monitor.record_step(_focus, action=_action, confidence=_conf)

                if result.stop_reason == "end_turn":
                    turn_messages.append(self.backend.make_assistant_message(result, raw_content))
                    final_text = result.text or "(no response)"

                    # One-time say() nudge (silence-control step 3): on a USER turn,
                    # if the model wrote text but never spoke, prompt it once to add
                    # a say(). It may still choose silence. Desire turns are exempt —
                    # autonomous turns decide their own voicing via the gates above.
                    if not say_used and not is_desire_turn and not say_nudge_used:
                        say_nudge_used = True
                        turn_messages.append(
                            self.backend.make_user_message(
                                "まだ声に出していない。必要なら say() で一言。"
                                "不要なら何もしなくてよい。"
                            )
                        )
                        continue

                    gate_method = getattr(self._meta_monitor, "gate_response", None)
                    gate: MetaGateDecision | None = None
                    if callable(gate_method):
                        maybe_gate = gate_method(
                            user_text=user_input,
                            candidate_response=final_text,
                            social_policy=social_policy,
                            last_error=self._last_tool_error,
                        )
                        if isinstance(maybe_gate, MetaGateDecision):
                            gate = maybe_gate
                    if gate is not None and gate.needs_repair and gate.repaired_response:
                        final_text = gate.repaired_response

                    continuation_status = "DONE"
                    status_match = re.search(
                        r"(?:^|\n)(DONE|CONTINUE:[^\n]+|DEFER:[^\n]+)\s*$", final_text
                    )
                    if status_match:
                        continuation_status = status_match.group(1)
                        final_text = final_text[: status_match.start(1)].rstrip() or "(no response)"
                    self._heartbeat.apply_status(continuation_status)

                    # Coherence gate: ask utility backend whether the response contains
                    # a logical error. Only fires once to avoid infinite loops.
                    _coherence_enabled = os.environ.get("FAMILIAR_COHERENCE_CHECK", "").strip() in (
                        "1",
                        "true",
                        "yes",
                    )
                    if _coherence_enabled and not getattr(self, "_coherence_retried", False):
                        violation = await self._check_response_coherence(final_text)
                        if violation:
                            self._coherence_retried = True
                            turn_messages.append(
                                self.backend.make_user_message(
                                    f"[SELF-CHECK] Your previous response has a problem: "
                                    f"{violation}. Please correct it and respond again."
                                )
                            )
                            say_used = False
                            continue

                    self._coherence_retried = False

                    # そのターンに自分がしたこと＝考えたこと（本文）と話したこと（say）。
                    # 区別せず両方を残す。本文は say() が出ると画面からは捨てられるが、
                    # 「考えたが言わなかったこと」として記憶には残す価値がある。
                    turn_record = "\n".join(
                        x for x in (*turn_thoughts, spoken_text) if x
                    )
                    if turn_record:
                        try:
                            self._mental_state_bus.append(mental_snapshot)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Failed to persist mental state snapshot: %s", exc)
                        # A（評価器 arousal）＝内容の novelty（外からの驚き。自発ターンは
                        # final_text へフォールバック）。keyword appraisal の arousal は使わない。
                        turn_arousal = await self._turn_arousal(user_input, turn_record)
                        self._spawn_background_task(
                            self._run_post_response_pipeline(
                                user_input=user_input,
                                final_text=turn_record,
                                camera_used=camera_used,
                                camera_image=camera_image,
                                observation_action_name=observation_action_name,
                                observation_action_input=observation_action_input,
                                companion_mood=companion_mood,
                                is_desire_turn=is_desire_turn,
                                desires=desires,
                                arousal=turn_arousal,
                                memories=memories,
                            ),
                            name="post-response-pipeline",
                        )
                        # Regenerate capabilities.yaml during rest turns (at most once per day).
                        if desire_name == "rest" and should_regenerate_manifest():
                            self._spawn_background_task(
                                self._regenerate_capability_manifest(),
                                name="capability-manifest-regen",
                            )

                        # Boost share_memory when note_to_share was actually called
                        # this turn (pending 登録経由の boost — Issue D).
                        if (
                            is_desire_turn
                            and desire_name
                            and not is_social_desire(desire_name)
                            and desires is not None
                        ):
                            _notes_n = getattr(
                                getattr(self, "_memory_tool", None),
                                "_notes_registered_this_turn", 0
                            )
                            if _notes_n > 0:
                                _share_boost = min(0.35, _notes_n * 0.15)
                                desires.boost("share_memory", _share_boost)
                                logger.debug(
                                    "note_to_share ×%d → share_memory +%.2f",
                                    _notes_n,
                                    _share_boost,
                                )

                    return final_text

                if result.stop_reason == "tool_use":
                    collected: list[tuple[str, str | None]] = []
                    for tc in result.tool_calls:
                        if tc.name == "see":
                            camera_used = True
                            if pending_view_action_name is not None:
                                observation_action_name = pending_view_action_name
                                observation_action_input = dict(pending_view_action_input or {})
                            else:
                                observation_action_name = "see"
                                observation_action_input = dict(tc.input)
                            pending_view_action_name = None
                            pending_view_action_input = None
                        elif tc.name in {"look", "walk"}:
                            pending_view_action_name = tc.name
                            pending_view_action_input = dict(tc.input)
                        # Capture whether say() was already used BEFORE updating say_used,
                        # so we can suppress duplicate audio in the same turn.
                        _is_duplicate_say = tc.name == "say" and say_used
                        if tc.name == "say":
                            if not _is_duplicate_say and not spoken_text:
                                spoken_text = str(tc.input.get("text", "")).strip()
                            say_used = True
                            non_say_streak = 0
                        else:
                            non_say_streak += 1
                        logger.info("Tool call: %s(%s)", tc.name, tc.input)
                        if on_action and not _is_duplicate_say:
                            on_action(tc.name, tc.input)

                        if _is_duplicate_say:
                            logger.warning(
                                "say() called again in same turn — duplicate audio suppressed"
                            )
                            text, image = "(duplicate say suppressed: already spoke this turn)", None
                        else:
                            timeout_s = self._tool_timeout_seconds(tc.name)
                            try:
                                text, image = await asyncio.wait_for(
                                    self._execute_tool(tc.name, tc.input),
                                    timeout=timeout_s,
                                )
                                self._last_tool_error = None
                                self._tool_failure_streak = 0
                            except asyncio.TimeoutError:
                                logger.warning("Tool %s timed out after %.1fs", tc.name, timeout_s)
                                text, image = (
                                    f"Tool timeout: {tc.name} exceeded {timeout_s:.1f}s.",
                                    None,
                                )
                                self._last_tool_error = text
                                self._tool_failure_streak += 1
                            except Exception as e:
                                logger.warning("Tool %s failed: %s", tc.name, e)
                                text, image = f"Tool error: {e}", None
                                self._last_tool_error = str(e)
                                self._tool_failure_streak += 1

                        if not _is_duplicate_say and (
                            tape_backend
                            and plan_ctx
                            and await check_plan_blocked(
                                tape_backend, plan_ctx, tc.name, tc.input, text
                            )
                        ):
                            logger.info("TAPE: plan blocked after %s, replanning...", tc.name)
                            replan = await generate_replan(
                                tape_backend, plan_ctx, tc.name, tc.input, text
                            )
                            if replan:
                                text = f"{text}\n\n[ADAPTIVE REPLAN] {replan}"
                                logger.info("TAPE replan: %s", replan[:80])

                        logger.info("Tool result: %s", text[:100])
                        if tc.name == "see" and image:
                            camera_image = image
                            _path_m = re.search(r"\(saved to ([^)]+)\)", text)
                            if _path_m:
                                asyncio.ensure_future(
                                    self._apply_face_hint(_path_m.group(1))
                                )
                        if image and on_image is not None:
                            on_image(image)
                        if on_tool_result is not None and not _is_duplicate_say:
                            on_tool_result(tc.name, tc.input, text)
                        collected.append((text, image))

                    turn_messages.append(self.backend.make_assistant_message(result, raw_content))
                    tool_msgs = self.backend.make_tool_results(result.tool_calls, collected)
                    turn_messages.append(tool_msgs)

                    if interrupt_queue is not None and not interrupt_queue.empty():
                        interrupts = self._drain_interrupt_queue(interrupt_queue)
                        if interrupts:
                            head = " / ".join(interrupts[:3])
                            if len(interrupts) > 3:
                                head += f" (+{len(interrupts) - 3} more)"
                            logger.debug("Consumed %d queued interrupts", len(interrupts))
                            turn_messages.append(
                                self.backend.make_user_message(
                                    f"[User interrupted x{len(interrupts)}]: {head}. "
                                    "Respond to this directly with say() now."
                                )
                            )
                            non_say_streak = 0

                    elif non_say_streak >= 2 and not say_used:
                        turn_messages.append(
                            self.backend.make_user_message(
                                "REMINDER: Writing text is silent. You MUST call say() to be heard. "
                                "Call say() NOW. Keep it to 1-2 sentences."
                            )
                        )
                        non_say_streak = 0

                    elif say_used and non_say_streak >= 2:
                        turn_messages.append(
                            self.backend.make_user_message(
                                "You already spoke. Stop exploring and end your turn now."
                            )
                        )
                        non_say_streak = 0

                    continue

                logger.warning("Unexpected stop_reason: %s", result.stop_reason)
                break

            logger.warning(
                "Reached max iterations (%d). Forcing final response.",
                turn_max_iterations,
            )
            turn_messages.append(
                self.backend.make_user_message(
                    "Please summarize what you found and provide your final answer now."
                )
            )
            result, _ = await self.backend.stream_turn(
                system=self._system_prompt(
                    morning_ctx=morning_ctx,
                    plan_ctx=plan_ctx,
                    continuity_ctx=continuity_ctx,
                    workspace_ctx=workspace_ctx,
                    mental_ctx=mental_ctx,
                ),
                messages=turn_messages,
                tools=[],
                max_tokens=turn_max_tokens,
                on_text=on_text,
            )
            return result.text or "(max iterations reached)"
        finally:
            self._restore_backend_after_turn(backend_turn_snapshot)
            if _internal_backend_saved is not None:
                # turn_messages はローカル変数なので self.messages は汚染されていない。復元不要。
                self.backend = _internal_backend_saved

    @property
    def stt(self) -> STTTool | None:
        """Speech-to-text tool, or None if not configured."""
        return self._stt

    def clear_history(self) -> None:
        """Clear conversation history (start fresh)."""
        self.messages = []
