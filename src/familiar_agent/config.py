"""Configuration management."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from ._i18n import _t

load_dotenv()


def _default_companion_name() -> str:
    return _t("default_companion_name")


def _resolve_float(field: str, env_name: str, default: float) -> float:
    """env > agent_state（内省の調整）> 既定 の順で決める（`config_overrides`）。

    `config_overrides` は `db` を使い、`db` は `config` を読むので、import は関数の中で行う。
    """
    try:
        from .config_overrides import resolve_float
        return resolve_float(field, env_name, default)
    except Exception:  # noqa: BLE001
        return _float_env(env_name, default)


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_value(*names: str, default: str = "") -> str:
    """Return the first present env var, preserving explicit empty strings."""
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return default


def _optional_int_env(*names: str) -> int | None:
    value = _env_value(*names, default="")
    if not value:
        return None
    return int(value)


def _bool_env(*names: str, default: bool = False) -> bool:
    value = _env_value(*names, default="")
    if not value:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class CameraConfig:
    host: str = field(
        default_factory=lambda: _env_value("CAMERA_HOST", "TAPO_CAMERA_HOST", default="")
    )
    username: str = field(
        default_factory=lambda: _env_value("CAMERA_USERNAME", "TAPO_USERNAME", default="admin")
    )
    password: str = field(
        default_factory=lambda: _env_value("CAMERA_PASSWORD", "TAPO_PASSWORD", default="")
    )
    port: int = field(
        default_factory=lambda: int(
            _env_value("CAMERA_ONVIF_PORT", "TAPO_ONVIF_PORT", default="2020")
        )
    )
    preview: bool = field(
        default_factory=lambda: os.environ.get("CAMERA_PREVIEW", "false").lower() == "true"
    )
    ptz_host_override: str = field(
        default_factory=lambda: _env_value("CAMERA_PTZ_HOST", default="")
    )
    ptz_username_override: str = field(
        default_factory=lambda: _env_value("CAMERA_PTZ_USERNAME", default="")
    )
    ptz_password_override: str = field(
        default_factory=lambda: _env_value("CAMERA_PTZ_PASSWORD", default="")
    )
    ptz_port_override: int | None = field(
        default_factory=lambda: _optional_int_env("CAMERA_PTZ_PORT")
    )
    # 定点（見回り・在席マップ・norm が共有する向き）。`名前:pan,tilt;…` の一行。
    # 値は ONVIF の正規化座標 [-1,1] で角度ではない。カメラのプリセットと合わせて使う。
    poses: str = field(default_factory=lambda: _env_value("CAMERA_POSES", default=""))
    # 同じ定点とみなす距離。pan 換算の角度で測るので 0.02 ＝ 3.4°（`poses.py`）。厳密一致を
    # 求めると絶対移動の誤差で毎回「移動中」になり定点ごとの「普通」が育たないが、実測の
    # 誤差は 1e-6 未満なので、この値でも到着は取りこぼさない。
    pose_tolerance: float = field(
        default_factory=lambda: _float_env("CAMERA_POSE_TOLERANCE", 0.02)
    )
    # 在席を確かめる間隔。カメラの動体イベントで起こされるので、短くする必要は無い。
    presence_interval_sec: float = field(
        default_factory=lambda: _float_env("CAMERA_PRESENCE_INTERVAL", 30.0)
    )
    # 動体イベントで起こされたときの下限間隔。動いているあいだイベントは毎秒何件も飛ぶので、
    # 素直に従うと 0.15 秒ごとに撮って YOLO を回すことになる（実機で観測）。動き始めの1件は
    # 待たせず、続けて飛んでくるぶんだけ間引く。
    presence_min_gap_sec: float = field(
        default_factory=lambda: _float_env("CAMERA_PRESENCE_MIN_GAP", 3.0)
    )
    # 滞留窓（`課題5` §I の在席 timeout）。静止している人は毎回検出されないので、
    # この時間の内側は居るものとして扱う。
    presence_window_sec: float = field(
        default_factory=lambda: _float_env("CAMERA_PRESENCE_WINDOW", 120.0)
    )

    def stream_url(self, stream: str = "stream1") -> str | int:
        """Build the RTSP or USB source URL — the single authoritative place.

        - host is a digit string or int → USB index (returns int)
        - host is empty → USB index 0
        - host contains '://' → pass through as-is
        - otherwise → rtsp://[user:pass@]host:554/{stream}
        """
        if isinstance(self.host, int) or (isinstance(self.host, str) and self.host.isdigit()):
            return int(self.host)
        if not self.host:
            return 0
        if "://" in self.host:
            return self.host
        auth = f"{self.username}:{self.password}@" if self.username and self.password else ""
        return f"rtsp://{auth}{self.host}:554/{stream}"

    def is_rtsp(self) -> bool:
        """True when the camera is an RTSP source (not USB/index)."""
        return bool(self.host) and not (isinstance(self.host, str) and self.host.isdigit())

    @property
    def ptz_host(self) -> str:
        return self.ptz_host_override or self.host

    @property
    def ptz_username(self) -> str:
        return self.ptz_username_override or self.username

    @property
    def ptz_password(self) -> str:
        return self.ptz_password_override or self.password

    @property
    def ptz_port(self) -> int:
        return self.ptz_port_override if self.ptz_port_override is not None else self.port


@dataclass
class MobilityConfig:
    api_region: str = field(default_factory=lambda: os.environ.get("TUYA_REGION", "us"))
    api_key: str = field(default_factory=lambda: os.environ.get("TUYA_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.environ.get("TUYA_API_SECRET", ""))
    device_id: str = field(default_factory=lambda: os.environ.get("TUYA_DEVICE_ID", ""))


@dataclass
class TTSConfig:
    elevenlabs_api_key: str = field(
        default_factory=lambda: os.environ.get("ELEVENLABS_API_KEY", "")
    )
    voice_id: str = field(
        default_factory=lambda: os.environ.get("ELEVENLABS_VOICE_ID", "cgSgspJ2msm6clMCkdW9")
    )
    go2rtc_url: str = field(
        default_factory=lambda: os.environ.get("GO2RTC_URL", "http://localhost:1984")
    )
    go2rtc_stream: str = field(default_factory=lambda: os.environ.get("GO2RTC_STREAM", "tapo_cam"))
    # Audio output routing: "local" = PC speaker only, "remote" = camera speaker only,
    # "both" = camera speaker + PC speaker simultaneously, "silent" = 出力しない（実機テスト用）。
    output: str = field(default_factory=lambda: os.environ.get("TTS_OUTPUT", "local"))
    # 合成をどこでやるか。`sbv2`＝Style-Bert-VITS2（ローカル・既定）／`elevenlabs`＝外部 API。
    # 設計（`計測・設定値 根拠台帳` §9）は SBV2 と声 jvnv-M2-jp を確定としている。**戻せる
    # ようにしてある**（SBV2 が動かないとき `TTS_ENGINE=elevenlabs` で従来へ）。
    engine: str = field(default_factory=lambda: os.environ.get("TTS_ENGINE", "sbv2"))
    # SBV2 は別プロセスの HTTP サーバーとして動かす。本体は Python 3.11・torch 2.10 だが
    # SBV2 は Python 3.12・torch 2.5・numpy 1.26.4 固定で、同じプロセスには載らない。
    sbv2_url: str = field(default_factory=lambda: os.environ.get("SBV2_URL", "http://127.0.0.1:5001"))
    sbv2_model: str = field(default_factory=lambda: os.environ.get("SBV2_MODEL", "jvnv-M2-jp"))
    # style は当面固定。PAD→(style, style_weight) の写像は次段（台帳 §9 が「別途設計」）。
    sbv2_style: str = field(default_factory=lambda: os.environ.get("SBV2_STYLE", "Neutral"))
    sbv2_weight: float = field(default_factory=lambda: _float_env("SBV2_WEIGHT", 1.0))
    sbv2_python: str = field(
        default_factory=lambda: os.environ.get("SBV2_PYTHON", "~/tts_eval/sbv2_env/bin/python")
    )
    sbv2_model_dir: str = field(
        default_factory=lambda: os.environ.get("SBV2_MODEL_DIR", "~/tts_eval/sbv2_models")
    )


@dataclass
class MemoryConfig:
    db_path: str = field(
        default_factory=lambda: os.environ.get(
            "MEMORY_DB_PATH",
            str(Path.home() / ".claude" / "memories"),
        )
    )
    # 拡散想起（[D-WR拡散想起]・(A)共起＋(B)エンティティ）。既定 on（DIFFUSE_RECALL=0 で無効化）。
    diffuse_recall: bool = field(default_factory=lambda: _bool_env("DIFFUSE_RECALL", default=True))
    diffuse_max_add: int = field(default_factory=lambda: _int_env("DIFFUSE_MAX_ADD", 4))
    diffuse_max_depth: int = field(default_factory=lambda: _int_env("DIFFUSE_MAX_DEPTH", 2))

    # 想起スコアのつまみ。既定値は課題5 v0.24（D 節＝合成／F 節＝新しさ）に一致させる。
    recall_half_life_days: float = field(  # HL=259200 秒（3日）
        default_factory=lambda: _float_env("RECALL_HALF_LIFE_DAYS", 3.0)
    )
    recall_time_floor: float = field(  # t_floor
        default_factory=lambda: _float_env("RECALL_TIME_FLOOR", 0.001)
    )
    # W に載せる想起件数。枠が少ないと、自己モデル文などが混じったとき本命が押し出される。
    recall_n: int = field(default_factory=lambda: _int_env("RECALL_N", 5))
    # 無関係排除の主たる足切り＝合成 final score の soft 床（生コサインではない）。
    # 0.05 起点（根拠台帳 §4・確定は5軸スコア分布の計測後）。
    recall_min_score: float = field(
        default_factory=lambda: _float_env("RECALL_MIN_SCORE", 0.05)
    )
    # 蒸留（記-a-ろ）の材料から外す新規性の下限。a0 が低い O は「既にある記憶と似ている
    # もの」で、実測でも同じ内容の繰り返しだった（`計測・設定値 根拠台帳`）。既定 0.47 は
    # 実測分布の p10。**内省が範囲内（0.20〜0.70）で調整できる**（`config_overrides`）。
    distill_min_a0: float = field(
        default_factory=lambda: _resolve_float(
            "MemoryConfig.distill_min_a0", "DISTILL_MIN_A0", 0.47
        )
    )
    # 合成床を課すときの候補過剰取得。採点後に絞ると n を割るため n×factor（上限 cap）取る。
    recall_overfetch_factor: int = field(
        default_factory=lambda: _int_env("RECALL_OVERFETCH_FACTOR", 3)
    )
    recall_overfetch_cap: int = field(
        default_factory=lambda: _int_env("RECALL_OVERFETCH_CAP", 20)
    )
    # 同じ内容の観測を続けて書かないための窓（秒）。0 で無効。
    dedup_window_secs: int = field(
        default_factory=lambda: _int_env("MEMORY_DEDUP_WINDOW_SECS", 30)
    )
    # r 軸の min-max 伸長係数。現行値では恒等（根拠台帳 v0.7 §3 の計測で決定）。
    recall_c_lo: float = field(  # c_lo
        default_factory=lambda: _float_env("RECALL_C_LO", 0.0)
    )
    recall_c_hi: float = field(  # c_hi
        default_factory=lambda: _float_env("RECALL_C_HI", 1.0)
    )
    # 重みプロファイル。w_r は関連ゲートの指数、残りは加算部 M の加重平均係数。
    # p 軸（w_p）は知覚待ちのため項ごと持たない。
    recall_w_r: float = field(default_factory=lambda: _float_env("RECALL_W_R", 1.0))
    recall_w_t: float = field(default_factory=lambda: _float_env("RECALL_W_T", 1.0))
    recall_w_e: float = field(default_factory=lambda: _float_env("RECALL_W_E", 1.0))
    recall_w_a: float = field(default_factory=lambda: _float_env("RECALL_W_A", 1.5))
    recall_w_p: float = field(default_factory=lambda: _float_env("RECALL_W_P", 1.0))
    # 在席者相関 p の候補集合拡張（slice-2）。在席他者視点でも候補を取り union する退避弁。
    recall_presence_expand: bool = field(
        default_factory=lambda: _bool_env("RECALL_PRESENCE_EXPAND", default=True)
    )
    # 感情一致 e のスケール σ（小=シビア／大=近縁も仲間）。λ_i は _emotion_match の既定。
    recall_emotion_sigma: float = field(
        default_factory=lambda: _float_env("RECALL_EMOTION_SIGMA", 1.0)
    )
    # 取込 novelty（内容の新規性）→ a0/A。AGENT_SELF 視点・self_model 除外・situated 近傍
    # K 件の平均コサインの裏返し（課題5 v0.26）。a0 = clip(w_n·novelty, 0, a0_cap)。
    novelty_k: int = field(default_factory=lambda: _int_env("NOVELTY_K", 7))
    novelty_w_n: float = field(default_factory=lambda: _float_env("NOVELTY_W_N", 1.5))
    novelty_default: float = field(default_factory=lambda: _float_env("NOVELTY_DEFAULT", 0.5))
    novelty_a0_cap: float = field(default_factory=lambda: _float_env("NOVELTY_A0_CAP", 1.5))
    # 自己認識 MI（W が空のときのデフォルト感情・外部 MI が入れば一員として参加）の重み。
    # 旧・activation 上限 C=2.0 の流用をやめ、支配しない薄い錨へ（emotion は REST が育てる）。
    self_mi_weight: float = field(
        default_factory=lambda: _float_env("SELF_MI_WEIGHT", 0.5)
    )


@dataclass
class PendingSpeechConfig:
    half_life_days: float = field(
        default_factory=lambda: _float_env("PENDING_SPEECH_HALF_LIFE_DAYS", 1.0)
    )
    floor: float = field(
        default_factory=lambda: _float_env("PENDING_SPEECH_FLOOR", 0.01)
    )
    expire_threshold: float = field(
        default_factory=lambda: _float_env("PENDING_SPEECH_EXPIRE_THRESHOLD", 0.1)
    )
    max_per_turn: int = field(
        default_factory=lambda: _int_env("PENDING_SPEECH_MAX", 2)
    )
    weight_content: float = field(
        default_factory=lambda: _float_env("ADDRESS_WEIGHT_CONTENT", 1.0)
    )
    weight_relation: float = field(
        default_factory=lambda: _float_env("ADDRESS_WEIGHT_RELATION", 1.0)
    )
    temperature: float = field(
        default_factory=lambda: _float_env("ADDRESS_TEMPERATURE", 1.0)
    )


@dataclass
class STTConfig:
    # Reuses ELEVENLABS_API_KEY — no separate key needed
    elevenlabs_api_key: str = field(
        default_factory=lambda: os.environ.get("ELEVENLABS_API_KEY", "")
    )
    language: str = field(default_factory=lambda: os.environ.get("STT_LANGUAGE", "ja"))


@dataclass
class CodingConfig:
    workdir: str = field(default_factory=lambda: os.environ.get("CODING_WORKDIR", ""))
    bash_enabled: bool = field(
        default_factory=lambda: os.environ.get("CODING_BASH", "false").lower() == "true"
    )


@dataclass
class RecognitionConfig:
    # 認識しきい値（cosine）＝「既知の人か」。face=ArcFace / voice=ECAPA。仮置き・実機で調整。
    face_threshold: float = field(
        default_factory=lambda: _float_env("FACE_THRESHOLD", 0.35)
    )
    voice_threshold: float = field(
        default_factory=lambda: _float_env("VOICE_THRESHOLD", 0.25)
    )
    # 自動切替しきい値（cosine）＝「話者を切り替えるほど確信あるか」。認識より少し上。
    # cosine 尺度が顔・声で違うため source 別に持つ（仮置き・実機で調整）。
    face_switch_threshold: float = field(
        default_factory=lambda: _float_env("FACE_SWITCH_THRESHOLD", 0.65)
    )
    voice_switch_threshold: float = field(
        default_factory=lambda: _float_env("VOICE_SWITCH_THRESHOLD", 0.35)
    )
    # 在席巡回（CameraPresenceWatcher）の周期（秒）。低頻度で誤確定・負荷を抑える。
    presence_interval_sec: float = field(
        default_factory=lambda: _float_env("PRESENCE_INTERVAL_SEC", 30.0)
    )
    # InsightFace のモデルパックと onnxruntime プロバイダ（CUDA→CPU フォールバック）。
    face_model: str = field(
        default_factory=lambda: os.environ.get("FACE_MODEL", "buffalo_l")
    )
    providers: str = field(
        default_factory=lambda: os.environ.get(
            "RECOGNITION_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider"
        )
    )

    def provider_list(self) -> list[str]:
        return [p.strip() for p in self.providers.split(",") if p.strip()]


@dataclass
class AgentConfig:
    # Agent display name shown in TUI
    # 名前は `ME.md`（「名前： …」）が持つ。env や設定画面からは与えない（正本を1つにする）。
    agent_name: str = "AI"
    # 名前として使える言葉の一覧（`ME.md` に読点で並べる）。沈黙依頼は、このどれかで
    # 呼ばれたときだけ受ける。表示用の主たる名前は `agent_name`。
    agent_names: list[str] = field(default_factory=list)

    # Name of the companion/user shown in TUI
    # 話者未識別時のデフォルトは中立ラベル「推定話者」（FAMILY.md 先頭メンバーへ derive しない）。
    # 顔/声/明示で確定したら set_active/apply_hint がそちらへ切り替える。
    companion_name: str = field(
        default_factory=lambda: os.environ.get("COMPANION_NAME", _t("gui_estimated_speaker"))
    )

    # 段階1スライス2：完了キュー経由の1反復1ツール連鎖の反復上限（暴走防止の安全弁）。
    # ネットの調べものは search（リンクが返る）→ fetch（本文を読む）→ 答える で最低3手。
    # 3 だと答える手が残らず、読めなかったときの取り直しの余地も無い（実機で観測）。
    event_max_iterations: int = field(
        default_factory=lambda: _int_env("EVENT_MAX_ITERATIONS", 5)
    )
    # 静穏時間＝**自分から**話しかけない時間帯（人への返事は掛からない）。出所は
    # 環境変数 → ここの既定の2段だけ（旧 schedule.conf・ROUTINES.md は撤去）。
    quiet_hours_start: int = field(default_factory=lambda: _int_env("QUIET_HOURS_START", 23))
    quiet_hours_end: int = field(default_factory=lambda: _int_env("QUIET_HOURS_END", 7))
    # 調べものがこの秒数を超えても結果が来なければ、「まだかかっている」を1回だけ知らせる。
    # 時計で定期的に起こすのではなく、**遅いという事実**が1回きりの起点になる。
    # 実測は search_deferred が平均2.5秒・最長22.1秒、fetch と recall は3〜4秒以内。
    lookup_slow_seconds: float = field(
        default_factory=lambda: _float_env("LOOKUP_SLOW_SECONDS", 5.0)
    )
    # 調停（軽量LLM）の時間切れ。届かなければフルへ倒す。普通の会話は 0.93〜1.10 秒だが、
    # 「黙って」と頼まれたときは **4.18 秒**かかると実測した（判断が重い）。2.0 秒では
    # 届かず、沈黙依頼が読まれないまま素通りしていた。0.8 秒の余裕を見て 5.0 とする。
    # 倒れたときは応答がこの秒数だけ遅れるので、伸ばしすぎない。
    arbiter_timeout_sec: float = field(
        default_factory=lambda: _float_env("ARBITER_TIMEOUT_SEC", 5.0)
    )
    # 同じ入力が続けて積まれるのを落とす窓（秒）。実機で1つの発言に2回答えた
    # （キューに2件入っていた）。入口はキーボードと音声の2つあり、どちらからでも積まれる。
    # **言い直しは弾かない**ので短く取る。0 にすると何も弾かない。
    input_dedupe_window_sec: float = field(
        default_factory=lambda: _float_env("INPUT_DEDUPE_WINDOW_SEC", 3.0)
    )
    # 「黙っていて」と頼まれてから、時間で解けるまでの長さ（分）。もう一つの解除は退室。
    # 「黙って」と頼まれたが長さを言われなかったときの既定。
    silence_minutes: int = field(default_factory=lambda: _int_env("SILENCE_MINUTES", 15))
    # 長さを言われたときの上限。超える指定は弾かずにここへ丸める（黙らないより意図に近い）。
    silence_max_minutes: int = field(
        default_factory=lambda: _int_env("SILENCE_MAX_MINUTES", 60)
    )
    # 完了 MI（調べた結果）の content 上限。取ってきた本文を切ると、表なら見出しだけが
    # 残って中身が消える。上限は埋め込みモデル bge-m3 の入力上限 8192 トークンに合わせる。
    # 1文字＝1トークンになる字もあるので、8192 *文字* なら常に 8192 トークン以下に収まり、
    # 埋め込みが後ろを落とさない。W に載る量は W の枠と溢れで別に制御される。
    completion_content_max: int = field(
        default_factory=lambda: _int_env("COMPLETION_CONTENT_MAX", 8192)
    )

    # Platform: "anthropic" | "gemini" | "openai" | "kimi" | "glm"
    platform: str = field(default_factory=lambda: os.environ.get("PLATFORM", "anthropic"))

    # Unified API key (used for whichever platform is selected).
    # Legacy ANTHROPIC_API_KEY is still accepted for backward compatibility.
    api_key: str = field(default_factory=lambda: _env_value("API_KEY", "ANTHROPIC_API_KEY"))

    # Model name — platform-specific defaults applied in create_backend().
    # Legacy ANTHROPIC_MODEL is still accepted for backward compatibility.
    model: str = field(default_factory=lambda: _env_value("MODEL", "ANTHROPIC_MODEL"))

    # OpenAI-compatible only: base URL and tool-calling mode
    # TOOLS_MODE: "native" = use function-calling API, "prompt" = inject into system prompt
    base_url: str = field(
        default_factory=lambda: os.environ.get("BASE_URL", "http://localhost:11434/v1")
    )
    tools_mode: str = field(default_factory=lambda: os.environ.get("TOOLS_MODE", "prompt"))

    # Thinking mode: "auto" | "adaptive" | "extended" | "disabled"
    # "auto" = adaptive for claude-sonnet-4/opus-4, disabled for others
    thinking_mode: str = field(default_factory=lambda: os.environ.get("THINKING_MODE", "auto"))

    # Budget tokens for "extended" thinking mode (ignored in "adaptive" / "disabled")
    thinking_budget: int = field(
        default_factory=lambda: int(os.environ.get("THINKING_BUDGET_TOKENS", "10000"))
    )

    # Effort level for adaptive thinking: "high" (default) | "medium" | "low" | "max"
    # "max" is Opus 4.6 only. Ignored unless THINKING_MODE=adaptive (or auto on supported models).
    thinking_effort: str = field(default_factory=lambda: os.environ.get("THINKING_EFFORT", "high"))

    realtime_stt: bool = field(default_factory=lambda: _bool_env("REALTIME_STT", default=False))

    # ── Utility backend (optional) ─────────────────────────────────────
    # Separate backend for non-conversation LLM calls (day summaries, emotion
    # inference, self-model updates, etc.).  Falls back to the main backend
    # when not configured.
    utility_platform: str = field(default_factory=lambda: os.environ.get("UTILITY_PLATFORM", ""))
    utility_api_key: str = field(default_factory=lambda: os.environ.get("UTILITY_API_KEY", ""))
    utility_model: str = field(default_factory=lambda: os.environ.get("UTILITY_MODEL", ""))

    # ── Scene backend (optional) ────────────────────────────────────────
    # Separate backend for scene entity extraction — cheaper/local model.
    # Falls back to utility backend (then main backend) when not configured.
    # For local VLM via Ollama: SCENE_PLATFORM=openai, SCENE_BASE_URL=http://localhost:11434/v1
    scene_platform: str = field(default_factory=lambda: os.environ.get("SCENE_PLATFORM", ""))
    scene_api_key: str = field(default_factory=lambda: os.environ.get("SCENE_API_KEY", ""))
    scene_model: str = field(default_factory=lambda: os.environ.get("SCENE_MODEL", ""))
    scene_base_url: str = field(default_factory=lambda: os.environ.get("SCENE_BASE_URL", ""))

    # ── Autonomous behavior ───────────────────────────────────────
    # Desire-driven idle turns are OFF by default.
    # Set FAMILIAR_AUTO=1 or FAMILIAR_AUTO_DESIRE=1 to enable.
    auto_desire: bool = field(
        default_factory=lambda: (
            os.environ.get("FAMILIAR_AUTO_DESIRE", "").strip().lower() in ("1", "true", "yes")
            or os.environ.get("FAMILIAR_AUTO", "").strip().lower() in ("1", "true", "yes")
        )
    )
    max_tokens: int = 4096
    camera: CameraConfig = field(default_factory=CameraConfig)
    mobility: MobilityConfig = field(default_factory=MobilityConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    coding: CodingConfig = field(default_factory=CodingConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)


@dataclass
class DriveConfig:
    """Drive 起動源の dynamics 定数（発火mood §2・課題5 B 由来）。

    値は設計の確定/仮値をそのまま既定にする（仮値の最終決定は課題8・実機）。
    b_i＝中立発火頻度のバイアス、c_*＝変調行列 C_ij（軸順 P, Pn, A, Dom）。
    """
    rate: float = 1.665e-2      # 全欲求共通の基準レート（/秒・課題5 B〔確定〕）
    p_t: float = 0.5            # T-tick 周期（秒・課題5 A）
    mult: float = 1.0          # 時間帯倍率（課題10・既定1.0）
    # 静穏時間の時間帯倍率は**軸ごとに違う**（設計式 mult_i(t)・#13）。
    #
    # 探索ほか4軸は 0.083。中立 mood（g_seeking = bias_seeking = 0.20）では探索が 0 から
    # Θ_fire へ至るまで 300 秒かかるので、3600 秒（1時間に1回程度）へ伸ばすと 300÷3600。
    #
    # REST だけは抑えず、逆に募らせる。設計（`設計詳細：発火・mood 機構` §82）が「REST の
    # 募りは別途バイアス＋時間帯倍率（夜高い）」と定めるためで、一律に掛けると正反対になる。
    # 3.0 は「深夜8時間（23〜7時）に必ず1回・2回は起きない」から出した。中立 mood での
    # 増分は毎秒 1.499e-5（rate × bias_rest）で、8時間ちょうど1回には 2.315 が要る。3.0 なら
    # 1.295 溜まって閾値を 30% 上回り、2回に必要な 4.63 には届かない。
    #
    # bond・safety・esteem は探索と同じ 0.083 のままにしてある。これらに適した値は
    # 検討していない（#13 は探索の頻度だけを見て決めた）。
    mult_quiet: float = 0.083
    mult_quiet_rest: float = 3.0
    # いま効いている REST の倍率（昼は `mult` と同じ 1.0）。静穏時間に
    # `effective_drive_cfg`（T 側）が `mult_quiet_rest` へ差し替える。
    mult_rest: float = 1.0
    learn: float = 1.0         # 学習倍率（課題10・既定1.0）
    epsilon: float = 0.001
    theta_fire: float = 1.0 - 0.001   # 発火閾値 Θ_fire = 1−ε
    discharge_q: float = 1.0 - 0.001  # 放電量 q = 1−ε（全放電）

    # バイアス b_i（中立時 g_{D,i}=b_i・0〜1・仮値）
    bias_seeking: float = 0.20
    bias_safety: float = 0.05
    bias_bond: float = 0.0056
    bias_esteem: float = 0.0014
    bias_rest: float = 0.0009

    # Slice 2b：新Drive発火で自発ターンを起こすか（既定 on＝新機能を前提・legacy DesireSystem と完全排他）
    autonomous: bool = field(default_factory=lambda: _bool_env("DRIVE5_AUTONOMOUS", default=True))

    # 自発ターンに同梱する drive5 スナップショットの定性ラベル帯（低<mid / 中 / 高≥high）。
    drive_level_mid: float = field(default_factory=lambda: _float_env("DRIVE_LEVEL_MID", 0.5))
    drive_level_high: float = field(default_factory=lambda: _float_env("DRIVE_LEVEL_HIGH", 0.75))

    # 案Y：ターン完了時に軽量LLMで満たされた drive を発火時と同じ全放電で沈静化するか（既定 on＝新機能を前提）。
    satisfy_llm: bool = field(default_factory=lambda: _bool_env("DRIVE5_SATISFY_LLM", default=True))
    # 充足判定LLM を回すゲートの PAD 距離しきい値（drive 値でなく E の動きで判定・上下両方向）。
    satisfy_gate_pad_dist: float = field(
        default_factory=lambda: _float_env("DRIVE5_SATISFY_PAD_DIST", 0.2)
    )

    # 発火→自発ターンの内声（[D-行動選択]・行動は指定せず主LLM が O と文脈から選ぶ）。
    # env で上書きして改善できる（VOICE_SEEKING 等）。
    voice_seeking: str = field(default_factory=lambda: os.environ.get(
        "VOICE_SEEKING",
        "探索したい気持ちが募っている。言葉にするだけで終えず、まず see・検索・look などの"
        "具体的な行動を1つ選んで実行し、その結果を踏まえて話す。探索する当てが本当に無いなら、"
        "「今は探索する当てがない」と理由まで結論づけて記憶に残す（曖昧に『気になることがある』"
        "と言って終えない）。"))
    voice_rest: str = field(default_factory=lambda: os.environ.get(
        "VOICE_REST",
        "休みたい気持ちが募っている。言葉にするだけで終えず、静かに落ち着くか活動を控えるかを"
        "具体的に選んで実行する。休む必要が特に無いなら、「今は休む必要はない」と理由まで"
        "結論づけて記憶に残す（曖昧に終えない）。"))
    voice_bond: str = field(default_factory=lambda: os.environ.get(
        "VOICE_BOND",
        "つながりたい気持ちが募っている。言葉にするだけで終えず、相手へ具体的に働きかけるか"
        "気にかける行動を1つ選んで実行する。今つながる相手や機会が無いなら、"
        "「今はつながる相手がいない」と理由まで結論づけて記憶に残す（曖昧に終えない）。"))
    voice_safety: str = field(default_factory=lambda: os.environ.get(
        "VOICE_SAFETY",
        "確かめたい・守りたい気持ちが募っている。言葉にするだけで終えず、see・look で見回るか"
        "状況を具体的に確認する行動を1つ選んで実行する。特に確かめることが無いなら、"
        "「今は確かめることはない」と理由まで結論づけて記憶に残す（曖昧に終えない）。"))
    voice_esteem: str = field(default_factory=lambda: os.environ.get(
        "VOICE_ESTEEM",
        "認められたい・役に立ちたい気持ちが募っている。言葉にするだけで終えず、何か示すか"
        "貢献する具体的な行動を1つ選んで実行する。今できる貢献が無いなら、"
        "「今は貢献できることがない」と理由まで結論づけて記憶に残す（曖昧に終えない）。"))

    # 変調行列 C_ij（各欲求の [P, Pn, A, Dom]・絶対値≤1.0・仮値）
    c_seeking: tuple[float, float, float, float] = (0.0, -0.5, 1.0, 0.4)
    c_safety: tuple[float, float, float, float] = (0.0, 0.75, 0.25, -1.0)
    c_bond: tuple[float, float, float, float] = (-1.0, 0.5, 0.0, 0.25)
    c_esteem: tuple[float, float, float, float] = (-0.2, 0.2, 1.0, -0.4)
    c_rest: tuple[float, float, float, float] = (-0.68, -0.68, -0.68, 0.0)

    def mult_for(self, axis: str) -> float:
        """その軸に掛ける時間帯倍率を返す。

        時計は見ない。静穏時間かどうかの判定は T が持ち（`loop/tonic.py` の
        `effective_drive_cfg`）、ここは差し替え済みの値を軸名で引くだけにする。
        """
        return self.mult_rest if axis == "rest" else self.mult
