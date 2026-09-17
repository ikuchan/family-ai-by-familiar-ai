"""Core agent loop - ReAct pattern with real-world tools."""

from __future__ import annotations
import asyncio
import contextlib
import logging
import math
import os
import re
import time
from collections.abc import Callable, Coroutine
from datetime import datetime
from pathlib import Path

from .core import parsing  # noqa: E402  ME.md/FAMILY.md/話者接頭辞の純粋パーサ
from .core.helpers import (  # noqa: F401,E402  切り出した純関数。内部利用＋既存の import 経路を保つ再輸出
    _call_optional_async,
    _noop_list,
    _noop_str,
    format_present_ctx,
)
from typing import Any

from .backends import create_backend, create_scene_backend, create_utility_backend
from .core.context_parts import Stance as _Stance
from .config import AgentConfig, DriveConfig
from .relationship import PersonRegistry
from .routines import quiet_hours_rule
from .io.aif import AIF, Nudge
from .store import clock
from .io.oif import MI, OIF, Cue, Recalled, View
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
from .tools.timer import TimerTool
from .tools.tts import TTSTool
from .loop.evaluator import Evaluator
from .loop.history import _flatten_history
from .mcp_client import CallResult, MCPClientManager, _resolve_config_path
from .capability_state import load_summary

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

# ── Thinking-mode switching ──────────────────────────────────────────────────
# Accepts:  /think [on|off|adaptive|disabled|status]
#           Natural-language standalone instructions (exact match)
_THINK_COMMAND_RE = re.compile(
    r"^/think(?:\s+(on|off|adaptive|disabled|status))?$",
    re.IGNORECASE,
)
# Exact natural-language phrases that toggle thinking (only when the ENTIRE
# message matches — avoids false positives mid-sentence)
_THINK_ON_EXACT = frozenset(
    {
        "深く考えて",
        "深く考えてください",
        "よく考えて",
        "じっくり考えて",
        "thinking on",
        "enable thinking",
    }
)
_THINK_OFF_EXACT = frozenset(
    {
        "考えなくていい",
        "考えなくていいです",
        "すぐに答えて",
        "シンプルに答えて",
        "thinking off",
        "disable thinking",
        "no thinking",
    }
)
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
# 区切りは半角空白のほか、全角空白と中黒（・）も通す。日本語入力のままスペースを押すと「・」に
# なり、`/speaker・` が普通の発話として記憶に残った（2026-09-15 実機）。
_SPEAKER_COMMAND_RE = re.compile(r"^/speaker(?:[\s　・]+(.*))?$", re.IGNORECASE)
_RELOAD_COMMAND_RE = re.compile(r"^/reload$", re.IGNORECASE)
_TIMER_COMMAND_RE = re.compile(r"^/timer[\s　・]+stop(?:[\s　・]+(.+))?$", re.IGNORECASE)

# Day summary prompt — condense a day's observations into a diary-like entry

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
        self._session_input_tokens: int = 0
        self._session_output_tokens: int = 0
        self._last_context_tokens: int = 0
        self._post_compact: bool = False
        self._coherence_retried: bool = False

        self._camera: CameraTool | None = None
        self._mobility: MobilityTool | None = None
        self._tts: TTSTool | None = None
        self._stt: STTTool | None = None
        self._me_md: str = self._load_me_md()  # loaded once; restart to pick up changes
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
        # **記憶ストア O との唯一の出入り口**（`設計図` ③-2・環-e-い）。書き込み・関係・
        # 埋め込みはここを通る（想起はまだ——器が違う）。**どの面へ書くかは `writer_id` が
        # 決める**ので、載せる記憶は基底でよい。読むときの面は `View.viewpoint` が言い、
        # **人ごとの実体は `pmm` が持つ**（1人につき1つ。口が作り直すと実体が増える）。
        self._oif = OIF(self._memory, for_person=self._pmm.get_memory_for)
        self._memory_tool = MemoryTool(self._pmm)
        self._pending_store = self._memory_tool._pending_store
        self._presence_sensor: PresenceSensor | None = None
        # 人検出（YOLO）。在席と `see` の即席の意味づけで共有（カメラが無ければ None）。
        self._person_detector: PersonDetector | None = None
        # 見えのエンコーダ（DINOv2）。起動時に温めるため参照を持つ。
        self._visual_encoder: VisualEncoder | None = None
        self._motion_events: MotionEventWatcher | None = None
        self._coding = CodingTool(config.coding)
        self._exploration = ExplorationTracker()
        self._scene: SceneTracker | None = None  # initialized after DB ready in _init_tools

        self._mcp: MCPClientManager | None = None
        self._persons = PersonRegistry(default_name=config.companion_name)
        self._prediction = PredictionEngine()
        # T との行き来はこの口へ集める（`設計図` ③-2 の4つの口）。I はループが
        # 立ち上がる前のターンでも Nudge を返すので、ここで持たせる。
        self._aif = AIF(None)
        self._schedule_rule = quiet_hours_rule()
        # タイマー（知-n）。器は表 `timers`、記録は O の `予定`。静穏時間・沈黙の依頼に掛かるものは
        # 確かめてから掛ける。止める口は道具 `cancel_timer` と命令 `/timer stop`。
        self._timer_tool = TimerTool(
            store=self._timer_store,
            oif=self._oif,
            speaker=lambda: self._persons.active_name if self._persons.active_is_explicit else "",
            quiet=lambda: self._schedule_rule,
            silence_active=self._silence_active_now,
            hush=self._hush_for_timer,
            unhush=self._unhush_timer,
            hush_enabled=bool(getattr(self.config, "timer_silence", True)),
            on_cancel=self._stop_timer_ring,
        )
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
            self._poses = await build_pose_registry(cam.poses, self._camera, cam.pose_tolerance)
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
        emotion_pad: "MoodPAD | None",  # 未測定でありうる（050）
        memories: "list[Recalled] | None",
        camera_used: bool,
    ) -> None:
        """ターン完了時、満たされた drive を軽量LLMで判定し発火時と同じ全放電で沈静化する。

        ゲートは drive 値でなく W/MI（memories）・E（PAD 距離）・行動から作る（鎮静対象を
        その値でゲートする循環を避ける）。既定 off。応答クリティカルパス外（本 pipeline 内）。
        """
        cfg = self._drive_config()
        if not cfg.satisfy_llm:
            return
        from .core.drive_satisfaction import (
            _AXES as _SATISFACTION_AXES,
        )
        from .core.drive_satisfaction import (
            apply_satisfaction,
            pad_distance,
            satisfaction_gate,
        )
        from .core.structured_ask import ask_subset

        # 中立からのズレ＝affect の大きさ（上下両方向）。**未測定なら 0**（050）——
        # 測れていないものを「中立だった」と読んで沈静化の判断に使わない。
        pad_move = 0.0 if emotion_pad is None else pad_distance(emotion_pad, MoodPAD())
        if not satisfaction_gate(
            memories_nonempty=bool(memories),
            pad_move=pad_move,
            action_used=camera_used,
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
        # 5軸のうち当てはまるものを列挙させる（出-d）。「無ければ none」と聞いているので、
        # **空集合は取れた答えであって失敗ではない**。読めなかったときは `None` が返る。
        # **パジュとして立つ。** 自分の欲求が満たされたかは、自分にしか分からない。
        axes = await ask_subset(
            self._utility_backend,
            prompt,
            choices=frozenset(_SATISFACTION_AXES),
            max_tokens=32,
            system=self._stance_context(_Stance.PAJU),
        )
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
        arousal: float = 0.0,
        memories: "list[Recalled] | None" = None,
        exchange_id: "int | None" = None,
        extra_cooccurring_ids: "list[str] | None" = None,
    ) -> None:
        """Persist and adapt after a reply without blocking that reply.

        exchange_id: 反復を閉じるときに同期で書いたやりとりの関係の id。会話要約を
            その末尾へ足す（無ければ足さない）。
        """
        if not final_text or final_text == "(no response)":
            return

        # 感情を PAD で1回評価し、ラベルは PAD から派生（W2b-2）。ターンの観測（生観測・
        # 会話 summary）にこの PAD を書き、派生ラベルは既存消費者へ渡す。
        # PAD は測れないことがある（050）。`emotion_pad` が None なら未測定で、
        # 気分では埋めない。A（高ぶり）は機械値なので常に返る。
        emotion_pad, emotion_a, emotion = await self._emotion_for_turn(final_text, arousal)
        self._update_mood(emotion)

        # mood を W トーンで nudge（mood-c）。W＝想起記憶（PAD, 根づき）＋現ターンの
        # 感情 E_cur（重み＝既定 a0=1.0）＋自己認識 MI フラット項（compute_n_pad が内包）。
        # 評価器の後に呼ぶ（E_cur を W に含めるため）。会話ターンのみ（memories が入力）。
        # W は口から `Recalled` で来る（環-e-い）。
        _nudge_items = [
            (r.mi.pad, r.groundedness)
            for r in (memories or [])
            # PAD が未測定の記憶は nudge の材料にしない（050）。測っていない中立で
            # 気分を引っ張ると、静かなターンほど気分が中立へ寄る。
            if r.mi.pad is not None
        ]
        # いまのターンの感情も**測れていなければ**材料にしない（050）。
        if emotion_pad is not None:
            _nudge_items.append((emotion_pad, 1.0))
        # T のレジスタは直接動かさない。行き来は AIF（自律機構接続）へ集める
        # （`設計図` ③-2 の4つの口）。
        self._aif.nudge(Nudge(items=_nudge_items))

        # そのターンで作った記憶 id（観察・会話）。共起の記録で W と結ぶ。
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
        )

        try:
            if camera_used:
                recent_obs = await self._oif.recall(
                    Cue(text=final_text[:200], direction="観察"), View(k=6)
                )
                past_scores = [r.fit for r in recent_obs[:3]]
                if past_scores:
                    avg_similarity = sum(past_scores) / len(past_scores)
                    novelty = 1.0 - avg_similarity
                else:
                    novelty = 0.8
                novelty = max(0.0, min(1.0, novelty))
                self._exploration.record_novelty(novelty)
                # 場面の更新と `観察` の書き込みはここから外した。この経路は
                # `loop/event_loop.py` の1箇所からしか来ず、そこは `camera_used=False`・
                # `camera_image=None`・`action_name=None` を渡すので、**どちらも一度も
                # 到達しない**。書いていた中身も `final_text`（自分の応答）で、同じ
                # テキストは `direction="発話"` の「自分が答えた：…」として既に残る。
                # 見た印は `InformationProcessing._write_seen_mark` が書く（定点名つき）。

            summary = await self._summarize_exchange(user_input, final_text)
            _conv_id = await self._oif.write(
                MI(
                    id="",
                    content=summary,
                    timestamp=None,
                    direction="会話",
                    emotion=emotion,
                    # **測っていなければ既定のまま**（050・中立で埋めると測ったのか
                    # 埋めたのかが見分けられなくなる）。
                    pad=emotion_pad if emotion_pad is not None else MoodPAD(),
                ),
                now=False,
                arousal=emotion_a,
                **self._conversation_perspective(),
            )
            _new_ids.append(_conv_id)

            # **答えの逐語は畳まない**（段 3）。畳むと想起の母集合から消え、残るのは
            # 軽量LLM の一文だけになる。細部のベクトルが無ければ、細部での近接は起きない。
            # 逐語と会話要約は、粒度の違う別々の記憶として並ぶ。

            # やりとりの関係は反復を閉じるときに同期で書かれている（`_finish`）。要約は
            # その末尾へ足す。関係を要約待ちにすると、閉じた直後の反復から直近のやりとりが
            # 見えない（2026-09-13 実機・入室の反復が `exchanges → 0件` を引いた）。
            if exchange_id and _conv_id:
                with contextlib.suppress(Exception):
                    self._oif.extend(exchange_id, [(_conv_id, "要約", None)])

            # 拡散想起の母集合：そのターンの W（想起 MI）と、そのターンに作った記憶を
            # **1つの共起**として記録する（新記憶↔W の接続・記録のみ・拡散は未接続）。
            self._record_cooccurrence(
                memories, list(_new_ids or []) + list(extra_cooccurring_ids or [])
            )

            # 人の印（`_last_human_at`）は入口（`push_utterance`）が付ける。ここでは書かない
            # （情-g・2026-09-15：自発ターンの cue も `user_input` に入るので、ここで書くと
            # ひとりの回数が毎ターン 0 へ戻る）。関係の追跡（旧表）は環-d で撤去。

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
            elevenlabs_model=tts.elevenlabs_model,
            sbv2_url=tts.sbv2_url,
            sbv2_style=tts.sbv2_style,
            sbv2_weight=tts.sbv2_weight,
        )

        cfg_path = _resolve_config_path()
        if cfg_path.exists():
            self._mcp = MCPClientManager(cfg_path)
        elif os.environ.get("MCP_CONFIG"):
            logger.warning("MCP_CONFIG points to non-existent file: %s", cfg_path)

        self._deferred_search = DeferredSearchTool(
            self._mcp_search, self._utility_backend, context=self._stance_context
        )
        self._deferred_fetch = DeferredFetchTool(self._mcp_search)

        stt_cfg = self.config.stt
        if stt_cfg.elevenlabs_api_key:
            cam = self.config.camera
            rtsp_url = str(cam.stream_url("stream1")) if cam.is_rtsp() else ""
            self._stt = STTTool(
                stt_cfg.elevenlabs_api_key,
                stt_cfg.language,
                rtsp_url,
                engine=stt_cfg.engine,
                stt_config=stt_cfg,
            )

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
            # 1つを在席とループ（`see` の即席の意味づけ）で共有する。重みの読込は 1 回。
            self._person_detector = PersonDetector()
            self._presence_sensor = PresenceSensor(
                camera=self._camera,
                poses_getter=self.poses,
                detector=self._person_detector,
                tolerance=cam_cfg.pose_tolerance,
                window_sec=cam_cfg.presence_window_sec,
                interval_sec=cam_cfg.presence_interval_sec,
                min_gap_sec=cam_cfg.presence_min_gap_sec,
            )
            # 見えの「普通」（`知覚在席` §3-4）。読めない環境でも在席（YOLO）は動き続ける。
            try:
                self._visual_encoder = VisualEncoder()
                self._presence_sensor.attach_visual_norm(
                    self._visual_encoder, PoseNormStore(_get_db().conn())
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

    async def _mcp_search(self, tool_name: str, tool_input: dict) -> "CallResult":
        """Route a search call through MCP, waiting for MCP init if needed.

        返りは `CallResult`（本文・画像・**ok**）。deferred の検索・取得が、道具の失敗を
        印で受け取って代わりの道具へ回すため（出-o）。
        """
        mcp_task = getattr(self, "_mcp_start_task", None)
        if mcp_task and not mcp_task.done():
            await mcp_task
        if self._mcp:
            return await self._mcp.call_result(tool_name, tool_input)
        return CallResult("MCP が利用できません。", None, False)

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

    def _stance_context(self, stance: "_Stance", *, with_rules: bool = False) -> "str | None":
        """立ち位置と文脈を組む（出-e）。**部品は正本から取り、控えを持たない。**

        人格とできることは `capability_state.load_summary()`、家族は `FAMILY.md`、
        規則は `loop.prompt.rules_section()` が正本である（チェッカーへは `rules_for_checker()`）。ここへ写しを置くと、正本が
        変わったときにここだけ古くなる。

        材料が欠けたら `None` を返す。`FAMILY.md` が無い機体や、自己認識をまだ生成して
        いない初回起動で**ターンごと落とさない**。そのときは立ち位置を渡さずに続く
        （いままでと同じ挙動）。
        """
        from .core.context_parts import build_context
        from .loop.prompt import rules_for_checker

        first_person = stance is _Stance.PAJU
        try:
            return build_context(
                stance=stance,
                self_understanding=(load_summary() or self._me_md) if first_person else "",
                family=self._family_md if first_person else "",
                # `with_rules` の呼び手は整合チェックだけ。渡すのは判定できる規則に絞った版
                # （出-n・`CHECKER_RULE_IDS`）。
                rules=(
                    rules_for_checker(allow_tts_tags=bool(self._tts and self._tts.understands_tags))
                    if with_rules
                    else ""
                ),
            ).stable
        except ValueError as e:
            logger.warning("立ち位置を組めなかったので渡さずに続ける: %s", e)
            return None

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
            ev = Evaluator(self._utility_backend, self.backend, context=self._stance_context)
            self.__dict__["_evaluator_obj"] = ev
        return ev

    def _active_memory(self) -> "ObservationMemory":
        """Return the current speaker's memory, or agent's own if no speaker is set."""
        return self._pmm.get_speaker_memory() or self._pmm.get_agent_memory()

    def _record_cooccurrence(
        self, memories: "list[Recalled] | None", new_ids: "list[str | None] | None" = None
    ) -> None:
        """そのターンの W（想起 MI）＋そのターンに作った記憶を1つの共起として記録する。

        新記憶↔W の接続を作る（拡散想起の母集合・記録のみ・挙動不変）。id は重複除去する。
        器は関係へ移した（段 5）。
        """
        from .store.relations import KIND_COOCCURRENCE, combine_cooccurring_ids

        mi_ids = combine_cooccurring_ids(memories, new_ids)
        if not mi_ids:
            return
        try:
            # **記憶の私的属性を掴まない**（環-e-い）。以前は `RelationStore(self._memory._ctx)`
            # と、記憶の内部構造を外から組み立てていた。共起は順序を持たない関係なので
            # 位置は `None`。
            self._oif.link(KIND_COOCCURRENCE, [(str(i), "項", None) for i in mi_ids])
        except Exception as e:  # noqa: BLE001
            logger.warning("共起の記録に失敗: %s", e)

    def _observation_perspective(self) -> dict:
        """知覚観察の面の材料（P1）。書き手＝エージェント自身、在席者は知覚から。

        列は 056 で落ちた。ここが渡すのは**面を立てる材料**で、書いた直後に `actor` と
        `present` の面になる。
        """
        return dict(
            writer_id=AGENT_SELF_ID,
            participants=self._pmm.get_present_ids(),
        )

    def _conversation_perspective(self) -> dict:
        """会話 summary の面の材料（P1）。書き手＝話者 floor DEFAULT・在席者。

        `scope` は 039 で列ごと落とした。誰との遣り取りかは `actor` と `present` の面が
        持っており、`scope` は同じことを別の語で重ねていた。`subject_id` は 056 で列ごと
        落とし、引数の受け渡しも撤去した（実在の人を指す 397 件は全件がその人の面を既に
        持っていた）。
        """
        speaker = self._pmm.current_speaker_id or DEFAULT_PERSON_ID
        return dict(
            writer_id=speaker,
            participants=self._pmm.get_present_ids(),
        )

    def _social_presence_permission(self) -> float:
        """**誰かがいれば** 1.0、部屋が空なら 0.0。社会的発話と deferred 配信の共通ゲート。

        「居るか」と「誰か」は別（知-h・2026-09-13）。**マイクは在席の証拠にしない**
        （2026-09-17：テレビ・物音・聞き違いを声として拾い、カメラが誰も見ていないのに
        「こんにちは」の書き起こしへ返事した）。数えるのは次の 3 つ。

        1. 在/不在の層が人を見ている（`PresenceSensor.room_occupied()`・YOLO・滞留窓）
        2. **自分が話してから** `presence_said_sec` 以内（話してよかった状態＝相手が居た、は
           しばらく続く。YOLO の見失いを跨ぐ）
        3. `/speaker` を打ってから同じ秒数以内（打った人はそこに居る）

        在席表（PMM・`/speaker`・顔照合）は「誰か」を言うもので、センサがある構成では
        居るかを決めない（`/speaker パパ` が永久に残り、カメラが 2 分「誰も居ない」でも自発が
        出た・同日 15:44）。センサが無い構成では在席表も数える（従来どおり）。
        """
        raw = getattr(getattr(self, "config", None), "presence_said_sec", None)
        window = float(raw) if isinstance(raw, (int, float)) and raw > 0 else 60.0
        now = time.time()

        def _within(attr: str) -> bool:
            at = getattr(self, attr, None)
            return isinstance(at, (int, float)) and (now - at) < window

        if _within("_last_said_at") or _within("_speaker_set_at"):
            return 1.0
        sensor = getattr(self, "_presence_sensor", None)
        if sensor is not None:
            with contextlib.suppress(Exception):
                if sensor.room_occupied() is True:
                    return 1.0
            return 0.0
        pmm = getattr(self, "_pmm", None)
        return 1.0 if pmm is not None and pmm.get_present_ids() else 0.0

    async def _nudge_seeking(self) -> None:
        """声がしたが応じられなかったので SEEKING を押し上げる（案ア・2026-09-17）。

        `step_drives` と同じ順（`db.lock` の中で読み・書き・commit）でスレッドへ逃がす。
        量は `DriveConfig.voice_nudge`（発火閾値の半分〔仮〕）。発火は T の tick に任せる。
        """
        from .config import DriveConfig
        from .core import drive_dynamics as dd
        from .drive_register import load_drives, save_drives

        def _work() -> tuple[float, float]:
            from .db import get_db

            cfg = DriveConfig()
            database = get_db()
            with database.lock:
                conn = database.conn()
                drives = load_drives(conn)
                nudged = dd.nudge(drives, "seeking", cfg.voice_nudge, cfg)
                save_drives(conn, nudged)
                conn.commit()
            return cfg.voice_nudge, nudged.seeking

        amount, now_value = await asyncio.to_thread(_work)
        logger.info(
            "drive 声がしたが誰も見えないので seeking を +%.2f（いま %.2f）", amount, now_value
        )

    def _stop_timer_ring(self) -> None:
        """鳴っているタイマーの音を止める（`cancel_timer`・`/timer stop` から）。"""
        ip = getattr(self, "_info_processing", None)
        dif = getattr(ip, "_dif", None)
        if dif is not None:
            with contextlib.suppress(Exception):
                dif.stop_ring()

    def _in_quiet_hours(self) -> bool:
        """Return True when the current time falls inside the scheduled quiet window.

        Safe default: if no schedule rule is configured, treat as NOT quiet.
        """
        rule = getattr(self, "_schedule_rule", None)
        return bool(rule is not None and rule.is_quiet())

    # Keywords that suggest the internal turn found something worth sharing.
    _INTERNAL_SHARE_PATTERNS: tuple[str, ...] = (
        "気になる",
        "面白い",
        "面白そう",
        "発見",
        "気づい",
        "思い出",
        "不思議",
        "見つけ",
        "変化",
        "新しい",
        "found",
        "discovered",
        "interesting",
        "noticed",
        "curious",
        "changed",
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

    def _load_me_md(self) -> str:
        """Load ME.md personality file if it exists（読む場所は `parsing.read_me_md`）。"""
        return parsing.read_me_md()

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
                self._persons.register(m["display_name"])
                logger.info(
                    "Family member registered: %s (display=%s)", m["name"], m["display_name"]
                )
            except Exception as exc:
                logger.warning("Could not register family member %s: %s", m["name"], exc)

    async def _emotion_for_turn(
        self, text: str, arousal: float
    ) -> "tuple[MoodPAD | None, float, str]":
        """評価器へ委譲（loop/evaluator.py）。テスト差し替え点として残す。"""
        return await self._evaluator.emotion_for_turn(text, arousal)

    async def _turn_arousal(self, user_input: str, final_text: str) -> float:
        """A（評価器 arousal）＝内容の新規性 novelty（課題5 v0.26）。

        外からの驚きを測るので user_input の novelty を使う。自発ターン（DMN・Drive
        発火）でユーザー入力が無いときは、エージェント自身の応答 final_text へフォールバック。
        """
        content = user_input if (user_input and user_input.strip()) else (final_text or "")
        return await self._oif.novelty(content)

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

    async def _anniversary_context(self) -> str | None:
        """Return a calendar-aware context string for today, or None if nothing notable.

        Surfaces "on this day" memories from past years and weekly/round milestones.
        Designed to be injected into morning reconstruction with high priority.
        """
        today = datetime.now().date()
        lines: list[str] = []

        # On-this-day memories (same month-day, past years)
        try:
            anniversaries = await self._oif.recall(Cue(on_month_day=(today.month, today.day)))
            for r in anniversaries[:2]:
                mem_date = clock.ts_to_date(r.mi.timestamp) if r.mi.timestamp else ""
                if r.mi.content and mem_date:
                    lines.append(f"[On this day]: {r.mi.content} ({mem_date})")
        except Exception:
            pass

        # Milestone: days since first memory
        try:
            first_date = (await self._oif.span()).earliest
            if first_date:
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

    async def _check_response_coherence(
        self, response: str, *, recent: str = "", facts: str = ""
    ) -> "str | None":
        """評価器へ委譲（loop/evaluator.py）。テスト差し替え点として残す。

        材料はループが集めて渡す。**会話履歴は渡さない**——`self.messages` は追記する
        箇所が1つも無く、いつも空である（出-f）。
        """
        return await self._evaluator.check_response_coherence(response, recent=recent, facts=facts)

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
        """Return True once the embedding model has finished loading.

        **ベクトル埋め込みは OIF の内側にある**（`設計図` ③-2）。記憶の面を直に見ると、
        設計の「埋め込みは口の内側」と食い違う（環-e-い）。
        """
        return self._oif.health().ready

    def embedding_failed(self) -> bool:
        """埋め込みモデルの読込に失敗したか（#10・致命）。記憶が死ぬので fail-fast する。

        使える状態かは口が答える（`OIF.health`・環-e-い）。
        """
        return self._oif.health().failed

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
        for watcher in (
            getattr(self, "_presence_sensor", None),
            getattr(self, "_motion_events", None),
        ):
            if watcher is not None:
                await watcher.start()

    async def interrupt(self, *, reason: str = "停止ボタン") -> None:
        """いま開いている求めを打ち切る（GUI の停止ボタン・環-j）。ループが無ければ何もしない。"""
        ip = getattr(self, "_info_processing", None)
        if ip is None:
            return
        await ip.abort_current(reason=reason)

    def set_output(self, on_text=None, on_action=None) -> None:
        """発話の表示先を登録する（アプリが起動時に渡す・人の発話を待たない）。

        自発の求め（メモ・タイマー・情動）は人が話しかける前にも起きる。出口を人の発話で
        初めて結ぶと、それまでの発話は画面にも音にも出ない（2026-09-15 実機 21:50）。
        """
        self._ensure_event_loop(on_text, on_action=on_action)

    def set_request_state_listener(self, listener) -> None:
        """求めが開いた／閉じたの通知先を登録する（GUI の停止ボタンが従う・環-j）。"""
        self._ensure_event_loop()
        self._info_processing.set_request_state_listener(listener)

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
            self._tonic = Tonic(
                self._info_processing, agent=self, presence=getattr(self, "_presence_sensor", None)
            )
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
        # ここから下は**一度でよいもの**（環-k・2026-09-14）。この関数は `_ensure_event_loop`
        # の末尾で呼ばれ、それは起動時と人の発話のたび（`run()`）に走る。MCP とワーカーは
        # 自分の状態で再入を止めるが、温めは止めておらず、発話ごとに YOLO のダミー推論が
        # 走っていた（実機 2026-09-13）。
        if getattr(self, "_services_primed", False):
            return
        self._services_primed = True
        # TTS の合成サーバー（SBV2）を起こす。モデルの読み込みに十数秒かかるので、最初の
        # 発話を待たせないよう起動時に投げておく（待たない・使う構成のときだけ）。
        with contextlib.suppress(Exception):
            from .tools.tts import ensure_sbv2_server

            tts_cfg = self.config.tts
            ensure_sbv2_server(tts_cfg, engine=tts_cfg.engine, output=tts_cfg.output)
        # 人検出（YOLO）と見えのエンコーダ（DINOv2）も起動時に温める。最初の see が
        # 読込込みで 5.3 秒かかり、「5 秒超え」のつなぎまで出た（2026-09-13 実機）。
        with contextlib.suppress(Exception):
            from .core.warmup import warm_models

            asyncio.ensure_future(
                warm_models(
                    detector=getattr(self, "_person_detector", None),
                    encoder=getattr(self, "_visual_encoder", None),
                )
            )
        # STT のモデル（faster-whisper）も起動時に読む。最初の書き起こしを待たせない。
        # 読み込みは GPU を触るのでスレッドへ逃がす（起動を塞がない）。
        with contextlib.suppress(Exception):
            from .tools.stt import ensure_whisper_model

            stt_cfg = self.config.stt
            if stt_cfg.engine == "whisper":
                asyncio.ensure_future(asyncio.to_thread(ensure_whisper_model, stt_cfg))

    async def _close_backends(self) -> None:
        """バックエンドのキャッシュを後始末する（出-i の呼び手）。

        **呼び手はここ1つだけである。** 心拍（定期的に `warm`）は入れないと決めた——
        温めると書き込みが丸ごと上乗せになり（1.063 ＋ 0.085 ＝ 1.148円 対 1.063円）、
        得られるのは1ターン目の書き込みが起動時へ前倒しになることだけである。

        **同じ実体を二度閉じない。** 軽量LLM や場面用を設定していなければ主LLM と同じ物を
        指す（`create_utility_backend(config) or self.backend`）。

        **閉じられなくても終了は続く。** キャッシュは速さと安さのためのもので、機能では
        ない。`anthropic` の `aclose` は手元の鍵を落とすだけだが、消さないと課金が続く
        バックエンドが将来入っても、閉じ口はここに揃う。
        """
        seen: set[int] = set()
        for backend in (self.backend, self._utility_backend, self._scene_backend):
            if backend is None or id(backend) in seen:
                continue
            seen.add(id(backend))
            aclose = getattr(backend, "aclose", None)
            if aclose is None:
                continue
            try:
                await aclose()
            except Exception as e:  # noqa: BLE001, PERF203
                logger.debug("バックエンドを閉じられなかった（続行する）: %s", e)

    async def close(self) -> None:
        """Clean up resources. Bounded by timeouts to avoid hanging on exit."""
        if self._camera:
            self._camera.close()

        # TTS の合成サーバー（SBV2）を止める。GPU を握り続けさせない。
        with contextlib.suppress(Exception):
            from .tools.tts import stop_sbv2_server

            stop_sbv2_server()

        await self._close_backends()

        # #11：T（自律機構）と I（情報処理機構）の常駐タスクを止める。
        tonic = getattr(self, "_tonic", None)
        if tonic is not None:
            await tonic.close()
        ip = getattr(self, "_info_processing", None)
        if ip is not None:
            await ip.close()

        await self._drain_background_tasks()

        # 終了時の日次要約と日記は撤去した（2026-09-14・記-a-ろ-は）。日次の畳み込みは
        # REST 内省の層 1（`loop/rest_fold.py`）が誰も居ない晩に行う。
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
        for _watcher in (
            getattr(self, "_presence_sensor", None),
            getattr(self, "_motion_events", None),
        ):
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

    async def _sync_pmm_speaker(self, name: str) -> None:
        """Set PersonMemoryManager speaker to match the name from PersonRegistry."""
        pmm = getattr(self, "_pmm", None)
        if pmm is None:
            return
        pid = pmm.find_person_id_by_name(name)
        if pid:
            await pmm.set_speaker(pid, source="text")

    def _handle_speaker_command(self, user_input: str) -> str | None:
        """/speaker [name] — set or show the active speaker for this session."""
        m = _SPEAKER_COMMAND_RE.match(user_input.strip())
        if m is None:
            return None
        name_arg = (m.group(1) or "").strip(" \t　・")
        if not name_arg:
            current = self._persons.active_name
            known = ", ".join(self._persons.known_names())
            return f"[現在の話者: {current}  既知: {known}]"
        self._persons.set_active(name_arg)
        self._speaker_set_at = time.time()  # 打った人はそこに居る（在席の証拠・60 秒）
        asyncio.ensure_future(self._sync_pmm_speaker(name_arg))
        return f"[話者を「{name_arg}」に切り替えました]"

    def _timer_store(self):
        """`TimerStore`（共有接続・`db.lock` の外で短く使う）。"""
        from .db import get_db
        from .store.timers import TimerStore

        return TimerStore(get_db().conn())

    def _silence_active_now(self) -> bool:
        """いま人に黙っているよう頼まれているか（タイマーを掛ける前の確認に使う・知-n）。

        タイマー由来の沈黙（`reason` が `timer:`）は数えない——動いているタイマーがあるだけで
        次のタイマーに「確かめて」が付くのは筋が違う。
        """
        try:
            from .silence_state import is_silenced, load_silence

            req = load_silence()
            if req is not None and req.reason.startswith("timer:"):
                return False
            present = {str(r.get("name") or "") for r in self._pmm.presence_status()}
            return is_silenced(req, present=present, now=time.time())
        except Exception:  # noqa: BLE001
            return False

    def _hush_for_timer(self, person: str, due, tid: int) -> None:
        """タイマーを掛けたら鳴るまで黙る（`TIMER_SILENCE`・2026-09-16）。"""
        try:
            from .silence_state import hush_for_timer, load_silence, save_silence

            current = load_silence()
            nxt = hush_for_timer(current, person=person, until=due.timestamp(), tid=tid)
            if nxt is not current:
                save_silence(nxt)
                logger.info(
                    "タイマーが鳴るまで黙る id=%d %s まで", tid, due.astimezone().strftime("%H:%M")
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("タイマーの沈黙を掛けられなかった: %s", e)

    def _unhush_timer(self, tid) -> None:
        """止めたタイマー由来の沈黙を解く。"""
        try:
            from .silence_state import clear_silence, load_silence, unhush_timer

            current = load_silence()
            if unhush_timer(current, tid=tid) is None and current is not None:
                clear_silence()
                logger.info("タイマーを止めたので沈黙を解く（%s）", tid)
        except Exception as e:  # noqa: BLE001
            logger.warning("タイマーの沈黙を解けなかった: %s", e)

    async def _handle_timer_command(self, user_input: str) -> str | None:
        """`/timer stop [id]`——LLM を通さずに止める（知-n・「途中で停められる」の非常口）。"""
        m = _TIMER_COMMAND_RE.match(user_input.strip())
        if m is None:
            return None
        target = (m.group(1) or "").strip(" \t　・") or "all"
        text, _ok = await self._timer_tool.call("cancel_timer", {"id": target})
        return text

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
        inner_voice: str = "",
        interrupt_queue=None,
    ) -> str:
        """人の発話で1ターン回す。

        中身はイベント駆動ループ（I と T）が持つ。スラッシュコマンドだけは LLM を
        呼ばずにここで返す。

        `on_image`・`on_phase`・`on_tool_result`・`inner_voice`・`interrupt_queue` は旧経路の
        引数で、いまはどれも使っていない。GUI と TUI が渡しているので受けるだけにしてある
        （呼び出し側の整理は #12a の後段）。`desires`・`desire_name` は環-d で落とした。
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

        # ── Timer command（/timer stop [id]・知-n・LLM を通さない非常口） ────────
        _timer_reply = await self._handle_timer_command(user_input)
        if _timer_reply is not None:
            if on_text:
                on_text(_timer_reply)
            return _timer_reply

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
        return await self._info_processing.push_utterance(user_input, on_text=on_text)

    @property
    def stt(self) -> STTTool | None:
        """Speech-to-text tool, or None if not configured."""
        return self._stt

    def clear_history(self) -> None:
        """Clear conversation history (start fresh)."""
        self.messages = []
