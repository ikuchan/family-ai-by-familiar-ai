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
    # "both" = camera speaker + PC speaker simultaneously.
    output: str = field(default_factory=lambda: os.environ.get("TTS_OUTPUT", "local"))


@dataclass
class MemoryConfig:
    db_path: str = field(
        default_factory=lambda: os.environ.get(
            "MEMORY_DB_PATH",
            str(Path.home() / ".claude" / "memories"),
        )
    )
    # 想起スコアのつまみ。既定値は課題5 v0.24（D 節＝合成／F 節＝新しさ）に一致させる。
    recall_half_life_days: float = field(  # HL=259200 秒（3日）
        default_factory=lambda: _float_env("RECALL_HALF_LIFE_DAYS", 3.0)
    )
    recall_time_floor: float = field(  # t_floor
        default_factory=lambda: _float_env("RECALL_TIME_FLOOR", 0.001)
    )
    recall_min_score: float = field(
        default_factory=lambda: _float_env("RECALL_MIN_SCORE", 0.0)
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
    # 感情一致 e のスケール σ（小=シビア／大=近縁も仲間）。λ_i は _emotion_match の既定。
    recall_emotion_sigma: float = field(
        default_factory=lambda: _float_env("RECALL_EMOTION_SIGMA", 1.0)
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
class AgentConfig:
    # Agent display name shown in TUI
    agent_name: str = field(default_factory=lambda: os.environ.get("AGENT_NAME", "AI"))

    # Name of the companion/user shown in TUI and ToM tool
    companion_name: str = field(
        default_factory=lambda: os.environ.get("COMPANION_NAME", _default_companion_name())
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
