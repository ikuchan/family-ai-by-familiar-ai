"""Core agent loop - ReAct pattern with real-world tools."""

from __future__ import annotations
import asyncio
import contextlib
import hashlib
import logging
import math
import os
import re
import time
from collections.abc import Callable, Coroutine
from datetime import datetime
from pathlib import Path

from .core import parsing  # noqa: E402  ME.md/FAMILY.md/話者接頭辞の純粋パーサ
from .core import brief_turn  # noqa: E402  brief-turn 判定・軽量返信モードのヒューリスティクス
from .core.helpers import (  # noqa: F401,E402  切り出した純関数。内部利用＋既存の import 経路を保つ再輸出
    _call_optional_async,
    _noop_list,
    _noop_str,
    format_present_ctx,
)
from typing import Any

from .backend import create_backend, create_scene_backend, create_utility_backend
from .concern_engine import ConcernEngine
from .config import AgentConfig, DriveConfig, MemoryConfig, PendingSpeechConfig
from .desires import DesireSystem, detect_worry_signal, is_social_desire
from .relationship import PersonRegistry, RelationshipTracker
from .routines import quiet_hours_rule
from .self_narrative import SelfNarrative
from .io.aif import AIF, Nudge
from .mood_register import MoodPAD
from .exploration import ExplorationTracker
from .scene import SceneTracker
from .poses import Pose, build_pose_registry
from .presence_sensor import PresenceSensor
from .prediction import PredictionEngine
from .memory_worker import MemoryJobWorker
from .tools.camera import CameraTool
from .tools.coding import CodingTool
from .tools.deferred_fetch import DeferredFetchTool
from .tools.deferred_search import DeferredSearchTool
from .tools.memory import MemoryTool, ObservationMemory
from .person_memory_manager import AGENT_SELF_ID, DEFAULT_PERSON_ID, PersonMemoryManager
from .recognition.motion_events import MotionEventWatcher
from .recognition.person_detector import PersonDetector
from .recognition.visual_encoder import VisualEncoder
from .store.pose_norms import PoseNormStore
from .tools.mobility import MobilityTool
from .tools.stt import STTTool
from .tools.tts import TTSTool
from ._i18n import _t
from .loop.evaluator import Evaluator
from .loop.history import _flatten_history
from .mcp_client import MCPClientManager, _resolve_config_path
from .capability_state import (
    build_generation_prompt,
    build_self_understanding_prompt,
    collect_manifest_context,
    load_manifest,
    save_manifest,
    filter_enabled,
    save_summary,
)

logger = logging.getLogger(__name__)








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
        # 名前の正本は `ME.md`（「名前： …」）。env や設定画面からは与えない。
        me_names = parsing.parse_me_names(self._me_md)
        if me_names:
            config.agent_names = me_names
            config.agent_name = me_names[0]
        self._memory = ObservationMemory()
        self._memory_worker = MemoryJobWorker(self._memory)
        self._pmm = PersonMemoryManager(self._memory)
        self._desires_ref: "DesireSystem | None" = None
        self._pmm.on_switch(self._on_pmm_speaker_switch)
        self._memory_tool = MemoryTool(self._pmm)
        self._pending_store = self._memory_tool._pending_store
        self._presence_sensor: PresenceSensor | None = None
        self._motion_events: MotionEventWatcher | None = None
        self._coding = CodingTool(config.coding)
        self._exploration = ExplorationTracker()
        self._scene: SceneTracker | None = None  # initialized after DB ready in _init_tools

        self._mcp: MCPClientManager | None = None
        self._persons = PersonRegistry(default_name=config.companion_name)
        # Property alias so all existing self._relationship.* calls continue to work.
        # They always address the currently active speaker's tracker.
        self._concerns = ConcernEngine()
        self._self_narrative = SelfNarrative()
        self._prediction = PredictionEngine()
        # T との行き来はこの口へ集める（`設計図` ③-2 の4つの口）。I はループが
        # 立ち上がる前のターンでも Nudge を返すので、ここで持たせる。
        self._aif = AIF(None)
        self._schedule_rule = quiet_hours_rule()
        self._last_tool_error: str | None = None
        self._tool_failure_streak: int = 0

        # Mood persistence (Phase 2 companion-likeness)
        self._mood: str = "neutral"
        self._mood_intensity: float = 0.0
        self._mood_set_at: float = time.time()

        # Deferred pre-response caches (computed in post-response, used next turn)

        # 定点。プリセットの読み出しに await が要るので、初めて要るときに一度だけ組む
        # （`_init_tools` は同期で、ここではまだカメラへ問い合わせられない）。
        self._poses: list[Pose] | None = None

        self._init_tools()

    async def poses(self) -> list[Pose]:
        """定点の一覧。在席マップ・norm・見回りが同じものを使う。"""
        if self._poses is None:
            cam = self.config.camera
            self._poses = await build_pose_registry(
                cam.poses, self._camera, cam.pose_tolerance
            )
            # `look` はこの一覧から選んで絶対移動する（道具の定義の enum にもなる）。
            if self._camera is not None:
                self._camera.set_poses(self._poses)
        return self._poses


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


    def _drive_config(self) -> DriveConfig:
        cfg = getattr(self, "_drive_cfg_cache", None)
        if cfg is None:
            cfg = self._drive_cfg_cache = DriveConfig()
        return cfg

    async def _maybe_discharge_satisfied_drives(
        self,
        *,
        user_input: str,
        final_text: str,
        emotion_pad: MoodPAD,
        memories: list[dict] | None,
        camera_used: bool,
        is_desire_turn: bool,
    ) -> None:
        """ターン完了時、満たされた drive を軽量LLMで判定し発火時と同じ全放電で沈静化する。

        ゲートは drive 値でなく W/MI（memories）・E（PAD 距離）・行動から作る（鎮静対象を
        その値でゲートする循環を避ける）。既定 off。応答クリティカルパス外（本 pipeline 内）。
        """
        cfg = self._drive_config()
        if not cfg.satisfy_llm:
            return
        from .core.drive_satisfaction import (
            apply_satisfaction,
            pad_distance,
            parse_satisfied_axes,
            satisfaction_gate,
        )

        pad_move = pad_distance(emotion_pad, MoodPAD())  # 中立からのズレ＝affect の大きさ（上下両方向）
        if not satisfaction_gate(
            memories_nonempty=bool(memories),
            pad_move=pad_move,
            action_used=camera_used or is_desire_turn,
            cfg=cfg,
        ):
            return

        prompt = (
            "次の対話ターンで、エージェント自身のどの欲求が『満たされた』かを判定してください。\n"
            "欲求は5つ：seeking（探索・好奇心）／rest（休息・鎮まり）／bond（つながり・絆）／"
            "safety（安全・安心）／esteem（承認・役立ち）。\n"
            "満たされたものだけを小文字の名前で列挙し、無ければ none とだけ答えてください。\n"
            f"[ユーザー] {user_input[:400]}\n[エージェント] {final_text[:400]}"
        )
        try:
            raw = await self._utility_backend.complete(prompt, max_tokens=32)
        except Exception as e:  # noqa: BLE001
            logger.warning("satisfaction check LLM failed: %s", e)
            return
        axes = parse_satisfied_axes(raw)
        if not axes:
            return

        from .db import get_db
        from .drive_register import load_drives, save_drives

        try:
            db = get_db()
            with db.lock:
                conn = db.conn()
                drives = load_drives(conn)
                drives = apply_satisfaction(drives, axes)
                save_drives(conn, drives)
                conn.commit()
            logger.info("Drive satisfied → discharged: %s", sorted(axes))
        except Exception as e:  # noqa: BLE001
            logger.warning("satisfaction discharge persist failed: %s", e)

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
        superseded_ids: list[str] | None = None,
        close_parent_id: str | None = None,
        extra_wr_ids: "list[str] | None" = None,
    ) -> None:
        """Persist and adapt after a reply without blocking that reply.

        superseded_ids: イベントループが消化した完了 O の id。ターンの観察保存後、その
        観察 id で supersede して想起から外す（完了結果の中間 O を残さない）。
        """
        if not final_text or final_text == "(no response)":
            return

        # 感情を PAD で1回評価し、ラベルは PAD から派生（W2b-2）。ターンの観測（生観測・
        # 会話 summary）にこの PAD を書き、派生ラベルは既存消費者へ渡す。
        emotion_pad, emotion = await self._emotion_for_turn(final_text, arousal)
        self._update_mood(emotion)

        # mood を W トーンで nudge（mood-c）。W＝想起記憶（PAD, 根づき）＋現ターンの
        # 感情 E_cur（重み＝既定 a0=1.0）＋自己認識 MI フラット項（compute_n_pad が内包）。
        # 評価器の後に呼ぶ（E_cur を W に含めるため）。会話ターンのみ（memories が入力）。
        _nudge_items = [
            (m["emotion_pad"], m["groundedness"])
            for m in (memories or [])
            if "emotion_pad" in m and "groundedness" in m
        ]
        _nudge_items.append((emotion_pad, 1.0))
        # T のレジスタは直接動かさない。行き来は AIF（自律機構接続）へ集める
        # （`設計図` ③-2 の4つの口）。
        self._aif.nudge(Nudge(items=_nudge_items))

        # そのターンで作った記憶 id（観察・会話）。WR 記録で W と共起させる。
        _new_ids: list[str | None] = []
        # 観察 O をここで書くのはやめた（上記）。None のまま残すのは、下の
        # supersede の宛先が `_obs_id or _conv_id` で会話へ落ちるためである。
        _obs_id: str | None = None

        # 案Y：満たされた drive を軽量LLMで判定し発火時と同じ全放電で沈静化（既定 off）。
        await self._maybe_discharge_satisfied_drives(
            user_input=user_input,
            final_text=final_text,
            emotion_pad=emotion_pad,
            memories=memories,
            camera_used=camera_used,
            is_desire_turn=is_desire_turn,
        )

        try:
            if camera_used:
                recent_obs = await self._memory.recall_async(
                    final_text[:200], n=6, kind="observation"
                )
                past_scores = [m.get("fit", 0.5) for m in recent_obs[:3]]
                if past_scores:
                    avg_similarity = sum(past_scores) / len(past_scores)
                    novelty = 1.0 - avg_similarity
                else:
                    novelty = 0.8
                novelty = max(0.0, min(1.0, novelty))
                self._exploration.record_novelty(novelty)
                if desires is not None:
                    desires.boost("look_around", novelty * 0.3)
                # 場面の更新と `観察` の書き込みはここから外した。この経路は
                # `loop/event_loop.py` の1箇所からしか来ず、そこは `camera_used=False`・
                # `camera_image=None`・`action_name=None` を渡すので、**どちらも一度も
                # 到達しない**。書いていた中身も `final_text`（自分の応答）で、同じ
                # テキストは `direction="発話"` の「自分が答えた：…」として既に残る。
                # 見た印は `InformationProcessing._write_seen_mark` が書く（定点名つき）。

            summary = await self._summarize_exchange(user_input, final_text)
            _conv_id, _ = await self._active_memory().save_async_with_id(
                summary,
                direction="会話",
                kind="conversation",
                emotion=emotion,
                dedupe_key=self._memory_dedupe_key("conversation", summary),
                materialize_now=False,
                emotion_pad=emotion_pad,
                **self._conversation_perspective(),
            )
            _new_ids.append(_conv_id)

            # イベントループが残したループ中 O を、このターンの記録で supersede（想起除外）。
            # カメラ分岐の中に置くと、カメラを使わないイベントループのターンでは一度も走らず
            # トリガ O が W に残り続ける（実機で観測）。観察が無ければ会話 O を宛先にする。
            _supersede_target = _obs_id or _conv_id
            if superseded_ids and _supersede_target:
                for _old in superseded_ids:
                    self._memory.mark_superseded(_old, _supersede_target)

            # 拡散想起の母集合：そのターンの W（想起 MI）と、そのターンに作った記憶を
            # 1つの WR として共起記録する（新記憶↔W の接続・記録のみ・拡散は未接続）。
            self._record_wr(memories, list(_new_ids or []) + list(extra_wr_ids or []))

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

            await self._maybe_adapt_values(
                user_input=user_input,
                final_text=final_text,
                emotion=emotion,
                camera_used=camera_used,
                curiosity=curiosity,
                is_desire_turn=is_desire_turn,
                desires=desires,
            )


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
            engine=tts.engine,
            sbv2_url=tts.sbv2_url,
            sbv2_style=tts.sbv2_style,
            sbv2_weight=tts.sbv2_weight,
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
            self._stt = STTTool(stt_cfg.elevenlabs_api_key, stt_cfg.language, rtsp_url,
                                engine=stt_cfg.engine, stt_config=stt_cfg)

        # World model: persistent scene entity tracker (Phase 1)
        # Shares the same PostgreSQL Database instance as ObservationMemory.
        from .db import get_db as _get_db

        try:
            self._scene = SceneTracker(_get_db())
        except Exception as exc:
            logger.warning("SceneTracker init failed: %s", exc)

        if self._camera:
            # 在/不在は YOLO で測る（登録が要らない）。誰かは PMM が必要時に解く。
            cam_cfg = self.config.camera
            self._presence_sensor = PresenceSensor(
                camera=self._camera,
                poses_getter=self.poses,
                detector=PersonDetector(),
                tolerance=cam_cfg.pose_tolerance,
                window_sec=cam_cfg.presence_window_sec,
                interval_sec=cam_cfg.presence_interval_sec,
                min_gap_sec=cam_cfg.presence_min_gap_sec,
            )
            # 見えの「普通」（`知覚在席` §3-4）。読めない環境でも在席（YOLO）は動き続ける。
            try:
                self._presence_sensor.attach_visual_norm(
                    VisualEncoder(), PoseNormStore(_get_db().conn())
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("見えの普通を用意できなかった（景色の驚きは出ない）: %s", exc)
            # カメラが「動いた」と言ってきたら、間隔を待たずに確かめる。静止している人は
            # 動体を出さないので、イベントだけでは足りず、間隔の確認と併せて使う。
            self._motion_events = MotionEventWatcher(
                self._connected_onvif,
                on_motion=self._presence_sensor.on_motion,
            )

        # Register family members from FAMILY.md into persons DB
        self._register_family_from_md()


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
            # `look` は定点へ絶対移動する。相対移動の積算を追う `ExplorationTracker` は
            # 前提が違うので繋がない（どこを見たかは「見た印」が O に残る）。器そのものの
            # 撤去は #12（旧系統の撤去）で、旧 `run()` のプロンプトごと落とす。
            return await self._camera.call(name, tool_input)
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


    # brief-turn 判定は core/brief_turn.py が持つ。既存の呼び出し口を保つ薄い委譲。
    _is_candidate_brief_turn = staticmethod(brief_turn.is_candidate_brief_turn)
    _should_use_brief_reply_mode = staticmethod(brief_turn.should_use_brief_reply_mode)


    _brief_reply_prompt = staticmethod(brief_turn.brief_reply_prompt)





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

    def _record_wr(
        self, memories: "list[dict] | None", new_ids: "list[str | None] | None" = None
    ) -> None:
        """そのターンの W（想起 MI）＋そのターンに作った記憶を1つの WR として共起記録する。

        新記憶↔W の接続を作る（拡散想起の母集合・記録のみ・挙動不変）。id は重複除去する。
        """
        from .wr_store import combine_wr_ids

        mi_ids = combine_wr_ids(memories, new_ids)
        if not mi_ids:
            return
        try:
            from .db import get_db
            from .wr_store import save_wr

            db = get_db()
            with db.lock:
                conn = db.conn()
                save_wr(conn, mi_ids)
                conn.commit()
        except Exception as e:  # noqa: BLE001
            logger.warning("WR record failed: %s", e)

    def _observation_perspective(self) -> dict:
        """知覚観察の視点列（P1）。書き手＝エージェント自身・情景観察・主体は話者 floor DEFAULT。"""
        speaker = self._pmm.current_speaker_id or DEFAULT_PERSON_ID
        return dict(
            writer_id=AGENT_SELF_ID,
            subject_id=speaker,
            participants=self._pmm.get_present_ids(),
        )

    def _conversation_perspective(self) -> dict:
        """会話 summary の視点列（P1）。書き手＝主体＝話者 floor DEFAULT・在席者。

        `scope` は 039 で列ごと落とした。誰との遣り取りかは writer_id と subject_id が
        持っており、`scope` は同じことを別の語で重ねていた。
        """
        speaker = self._pmm.current_speaker_id or DEFAULT_PERSON_ID
        return dict(
            writer_id=speaker,
            subject_id=speaker,
            participants=self._pmm.get_present_ids(),
        )

    def _social_presence_permission(self) -> float:
        """誰か居れば 1.0、部屋が空なら 0.0。社会的発話と deferred 配信の共通ゲート。

        在席の証拠は2つあり、**どちらかが立てば在席**とする。カメラの有無で根拠を
        切り替えない。

        1. 顔が検出されている（PMM に在席者が居る）
        2. 直近5分以内に人が話しかけてきた（対話は在席の直接的な証拠）

        以前はカメラが有効なとき 1 だけを見て確定しており、目の前で人が話しかけていても
        顔が識別されなければ「誰も居ない」と判定して返事まで保留にしていた（実機で観測）。

        **暫定である点の明示**：ここで使う `get_present_ids()` は InsightFace が埋める
        **人物 id（誰か＝identity）**であって、設計が定める**在席（在/不在）**そのものでは
        ない。正本は二層に分けており（在/不在＝T(G)・YOLO で連続／誰か＝I・InsightFace で
        必要時）、identity を presence の代わりに使うのは暫定にすぎない。**二層の分離は
        残課題 #8（在席系の精緻化）で扱う**。
        """
        pmm = getattr(self, "_pmm", None)
        if pmm is not None and pmm.get_present_ids():
            return 1.0
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


    async def _emotion_for_turn(self, text: str, arousal: float) -> tuple[MoodPAD, str]:
        """評価器へ委譲（loop/evaluator.py）。テスト差し替え点として残す。"""
        return await self._evaluator.emotion_for_turn(text, arousal)


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
                    min_score=MemoryConfig().recall_min_score,
                )
                collected += assoc
            except Exception:
                pass

        seen: set[str] = set()
        merged: list[dict] = []
        for m in sorted(collected, key=lambda x: x.get("fit", 0.0), reverse=True):
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
            # 自己認識は1枚（案B）。ME.md（人が書いた人格）を素材に、実装から導いた
            # 「できること」を足す。有効条件は実際に評価する（条件つき≠有効）。
            prompt = build_self_understanding_prompt(
                me_md=getattr(self, "_me_md", ""),
                manifest=filter_enabled(manifest),
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

    def embedding_failed(self) -> bool:
        """埋め込みモデルの読込に失敗したか（#10・致命）。記憶が死ぬので fail-fast する。"""
        return self._memory.embedding_failed()

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
                recent = await self._memory.recall_async("", n=5)
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

    async def _connected_onvif(self):
        """ONVIF を繋いでから返す。動体イベントの購読先。

        `_cam_onvif` は `_ensure_connected()` を呼ぶまで None なので、素の属性を渡すと
        購読が「カメラが無い構成」とみなして静かに終わる（実機で観測）。
        """
        camera = self._camera
        if camera is None or not await camera._ensure_connected():
            return None
        return camera._cam_onvif

    async def start_autonomy(self) -> None:
        """起動したら自律の側を回し始める（人の発話を待たない）。

        `感情ループ全体像` は「起動源は Drive の時間蓄積と発火」と定めるが、実装では I も T も
        在席センサも動体イベントも `run()` の中、しかも人の入力があるときにしか立たなかった。
        起動しても、話しかけるまで何ひとつ回っていない。同じ根から3つの症状が出ていた。

        - `/speaker` を最初に打つと入室イベントが立たない（T がまだ無い）
        - 在席が「連続」にならない（`知覚在席` §3-2 は G（T 側・連続）と定める）
        - 保留していた発話を配る起点（在席がゼロから立ち上がる瞬間）が来ない

        GUI と CUI の両方の入口から呼ぶ。何度呼んでも二重には立たない。
        """
        # 定点を先に読む。`look` の道具定義（enum）がこれを要るので、遅れると最初の
        # 反復で「見に行く」が選べない。
        with contextlib.suppress(Exception):
            await self.poses()
        self._ensure_event_loop()
        for watcher in (getattr(self, "_presence_sensor", None),
                        getattr(self, "_motion_events", None)):
            if watcher is not None:
                await watcher.start()

    def _ensure_event_loop(self, on_text=None, on_action=None) -> None:
        """I（情報処理機構）と T（自律機構）を用意する。

        T は時計を持つ唯一の側で、drive を進めて発火を QA へ積む。
        """
        from .loop.event_loop import InformationProcessing
        from .loop.tonic import Tonic

        if getattr(self, "_info_processing", None) is None:
            self._info_processing = InformationProcessing(self)
        if on_text is not None or on_action is not None:
            self._info_processing.set_output(on_text, on_action=on_action)
        self._info_processing.start()
        if getattr(self, "_tonic", None) is None:
            self._tonic = Tonic(self._info_processing, agent=self,
                                presence=getattr(self, "_presence_sensor", None))
        self._tonic.start()
        # RH（資源ハンドラ）の完了を QC へ渡す。
        ip = self._info_processing
        for tool in (self._deferred_search, self._deferred_fetch):
            with contextlib.suppress(Exception):
                tool.set_completion_sink(ip.push_completion)
        # MCP とメモリワーカーの起動は run() の中にあり、イベントループの分岐は run() の
        # 先頭で return するため到達しなかった。結果 MCP のツールが登録されず、検索が
        # 「tool not found」で即失敗していた（実機で観測）。ここでも起こす。
        self._start_background_services()

    def _start_background_services(self) -> None:
        """MCP とメモリワーカーを起こす（未起動なら）。run() とイベントループの両方から呼ぶ。"""
        mcp = getattr(self, "_mcp", None)
        if mcp is not None and not mcp.is_started:
            task = getattr(self, "_mcp_start_task", None)
            if task is None or task.done():
                self._mcp_start_task = asyncio.ensure_future(mcp.start())
        worker = getattr(self, "_memory_worker", None)
        if worker is not None and not worker.is_running:
            asyncio.ensure_future(worker.start())
        # TTS の合成サーバー（SBV2）を起こす。モデルの読み込みに十数秒かかるので、最初の
        # 発話を待たせないよう起動時に投げておく（待たない・使う構成のときだけ）。
        with contextlib.suppress(Exception):
            from .tools.tts import ensure_sbv2_server

            tts_cfg = self.config.tts
            ensure_sbv2_server(tts_cfg, engine=tts_cfg.engine, output=tts_cfg.output)
        # STT のモデル（faster-whisper）も起動時に読む。最初の書き起こしを待たせない。
        # 読み込みは GPU を触るのでスレッドへ逃がす（起動を塞がない）。
        with contextlib.suppress(Exception):
            from .tools.stt import ensure_whisper_model

            stt_cfg = self.config.stt
            if stt_cfg.engine == "whisper":
                asyncio.ensure_future(asyncio.to_thread(ensure_whisper_model, stt_cfg))

    async def close(self) -> None:
        """Clean up resources. Bounded by timeouts to avoid hanging on exit."""
        if self._camera:
            self._camera.close()

        # TTS の合成サーバー（SBV2）を止める。GPU を握り続けさせない。
        with contextlib.suppress(Exception):
            from .tools.tts import stop_sbv2_server

            stop_sbv2_server()

        heartbeat = getattr(self, "_cache_heartbeat_task", None)
        if heartbeat and not heartbeat.done():
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

        # #11：T（自律機構）と I（情報処理機構）の常駐タスクを止める。
        tonic = getattr(self, "_tonic", None)
        if tonic is not None:
            await tonic.close()
        ip = getattr(self, "_info_processing", None)
        if ip is not None:
            await ip.close()

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
        for _watcher in (getattr(self, "_presence_sensor", None),
                         getattr(self, "_motion_events", None)):
            if _watcher is not None:
                try:
                    await asyncio.wait_for(_watcher.stop(), timeout=1.0)
                except (asyncio.TimeoutError, Exception):  # noqa: BLE001, PERF203
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
            me_names = parsing.parse_me_names(self._me_md)
            if me_names:
                self.config.agent_names = me_names
                self.config.agent_name = me_names[0]
        else:
            lines.append("• ME.md 変更なし")
        if self._family_md != old_family:
            lines.append("• FAMILY.md を更新しました")
            self._register_family_from_md()
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
        """人の発話で1ターン回す。

        中身はイベント駆動ループ（I と T）が持つ。スラッシュコマンドだけは LLM を
        呼ばずにここで返す。

        `on_image`・`on_phase`・`on_tool_result`・`desires`・`inner_voice`・`desire_name`・
        `interrupt_queue` は旧経路の引数で、いまはどれも使っていない。GUI と TUI が
        渡しているので受けるだけにしてある（呼び出し側の整理は #12a の後段）。
        """
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

        if not user_input:
            return ""
        # I（情報処理機構）と T（自律機構）を用意し、LPM の反復を回す。
        # GUI は「発話は on_action("say") で来る」前提で作られている（素テキストは
        # say の前の途中経過としてしか扱わず、say が出たら捨てる）。渡さないと GUI に
        # 何も表示されない（実機で観測）。
        self._ensure_event_loop(on_text, on_action)
        return await self._info_processing.run_iteration(user_input, on_text=on_text)

    @property
    def stt(self) -> STTTool | None:
        """Speech-to-text tool, or None if not configured."""
        return self._stt

    def clear_history(self) -> None:
        """Clear conversation history (start fresh)."""
        self.messages = []
