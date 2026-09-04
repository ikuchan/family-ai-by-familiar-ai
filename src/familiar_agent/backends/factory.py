"""どのバックエンドを使うかを決める（出-d-は）。

`PLATFORM` と `MODEL` から実体を選ぶ。主LLM・軽量LLM・情景の意味づけで別の口がある。
"""

from __future__ import annotations

import logging
import os
import shlex
from typing import TYPE_CHECKING

from .anthropic import AnthropicBackend
from .cli import CLIBackend
from .gemini import GeminiBackend
from .glm import GLMBackend
from .kimi import KimiBackend
from .openai_compat import OpenAICompatibleBackend

if TYPE_CHECKING:
    from ..config import AgentConfig

logger = logging.getLogger(__name__)

def create_backend(
    config: "AgentConfig",
) -> (
    AnthropicBackend
    | OpenAICompatibleBackend
    | KimiBackend
    | GLMBackend
    | GeminiBackend
    | CLIBackend
):
    """Factory: pick backend based on PLATFORM env var / config.

    Supported values for PLATFORM:
      anthropic  — Anthropic Claude (default)
      gemini     — Google Gemini via native google-genai SDK
      openai     — OpenAI API (or compatible via BASE_URL)
      kimi       — Moonshot AI Kimi K2.5 (api.moonshot.ai/v1)
      glm        — Z.AI GLM API (api.z.ai/api/paas/v4); set ZAI_API_KEY
      cli        — any CLI LLM tool via stdin/stdout (MODEL = the command)
                   e.g. MODEL="claude -p"  or  MODEL="ollama run gemma3:27b"
    """
    if config.platform == "gemini":
        model = config.model or "gemini-2.5-flash"
        logger.info("Using Gemini backend: %s", model)
        return GeminiBackend(api_key=config.api_key, model=model)
    if config.platform == "openai":
        model = config.model or "gpt-4o-mini"
        # If BASE_URL not explicitly set, use the real OpenAI endpoint
        base_url = config.base_url
        if not os.environ.get("BASE_URL"):
            base_url = "https://api.openai.com/v1"
        # Default to "prompt" for local/compatible endpoints; "native" only for real OpenAI.
        # Local model servers (LM Studio, Ollama, vllm, etc.) often hang or timeout when
        # they receive the `tools` parameter without proper support — causing Request timed out.
        is_real_openai = "api.openai.com" in base_url
        tools_mode = (
            config.tools_mode
            if os.environ.get("TOOLS_MODE")
            else ("native" if is_real_openai else "prompt")
        )
        logger.info(
            "Using OpenAI backend: %s @ %s (tools=%s)",
            model,
            base_url,
            tools_mode,
        )
        return OpenAICompatibleBackend(
            api_key=config.api_key,
            model=model,
            base_url=base_url,
            tools_mode=tools_mode,
        )
    if config.platform == "kimi":
        # Moonshot AI Kimi K2.5 — needs its own backend to handle reasoning_content
        # See: https://platform.moonshot.ai / https://github.com/MoonshotAI/Kimi-K2.5
        model = config.model or "kimi-k2.5"
        logger.info("Using Kimi backend: %s", model)
        return KimiBackend(api_key=config.api_key, model=model)
    if config.platform == "glm":
        # Z.AI GLM — OpenAI-compatible, native tool calling
        model = config.model or "glm-4.6v"
        logger.info("Using GLM backend: %s", model)
        return GLMBackend(api_key=config.api_key, model=model)
    if config.platform == "cli":
        raw_cmd = config.model.strip() if config.model else "claude -p {}"
        cmd = shlex.split(raw_cmd)
        logger.info("Using CLI backend: %s", " ".join(cmd))
        return CLIBackend(cmd)
    model = config.model or "claude-haiku-4-5-20251001"
    logger.info("Using Anthropic backend: %s", model)
    return AnthropicBackend(
        api_key=config.api_key,
        model=model,
        thinking_mode=config.thinking_mode,
        thinking_budget=config.thinking_budget,
        thinking_effort=config.thinking_effort,
    )


def create_utility_backend(
    config: "AgentConfig",
) -> AnthropicBackend | OpenAICompatibleBackend | KimiBackend | GLMBackend | GeminiBackend | None:
    """Create a separate backend for utility LLM calls (summaries, emotion, etc.).

    Returns None if UTILITY_PLATFORM is not configured — caller should
    fall back to the main conversation backend.
    """
    if not config.utility_platform or not config.utility_api_key:
        return None

    platform = config.utility_platform
    api_key = config.utility_api_key
    model = config.utility_model

    if platform == "anthropic":
        model = model or "claude-haiku-4-5-20251001"
        logger.info("Using Anthropic utility backend: %s", model)
        return AnthropicBackend(api_key=api_key, model=model, thinking_mode="disabled")
    if platform == "gemini":
        model = model or "gemini-2.5-flash"
        logger.info("Using Gemini utility backend: %s", model)
        return GeminiBackend(api_key=api_key, model=model)
    if platform == "kimi":
        model = model or "kimi-k2.5"
        logger.info("Using Kimi utility backend: %s", model)
        return KimiBackend(api_key=api_key, model=model)
    if platform == "glm":
        model = model or "glm-4.6v"
        logger.info("Using GLM utility backend: %s", model)
        return GLMBackend(api_key=api_key, model=model)
    if platform == "openai":
        model = model or "gpt-4o-mini"
        logger.info("Using OpenAI utility backend: %s", model)
        return OpenAICompatibleBackend(
            api_key=api_key, model=model, base_url="https://api.openai.com/v1"
        )

    logger.warning("Unknown UTILITY_PLATFORM: %s, falling back to main backend", platform)
    return None


def create_scene_backend(
    config: "AgentConfig",
) -> AnthropicBackend | OpenAICompatibleBackend | KimiBackend | GLMBackend | GeminiBackend | None:
    """Create a separate backend for scene entity extraction (cheap/local model).

    Returns None if SCENE_PLATFORM is not configured — caller should fall back
    to the utility backend or main backend.
    """
    if not config.scene_platform or not config.scene_api_key:
        return None

    platform = config.scene_platform
    api_key = config.scene_api_key
    model = config.scene_model

    if platform == "anthropic":
        model = model or "claude-haiku-4-5-20251001"
        logger.info("Using Anthropic scene backend: %s", model)
        return AnthropicBackend(api_key=api_key, model=model, thinking_mode="disabled")
    if platform == "gemini":
        model = model or "gemini-2.5-flash"
        logger.info("Using Gemini scene backend: %s", model)
        return GeminiBackend(api_key=api_key, model=model)
    if platform == "kimi":
        model = model or "kimi-k2.5"
        logger.info("Using Kimi scene backend: %s", model)
        return KimiBackend(api_key=api_key, model=model)
    if platform == "glm":
        model = model or "glm-4.6v"
        logger.info("Using GLM scene backend: %s", model)
        return GLMBackend(api_key=api_key, model=model)
    if platform == "openai":
        model = model or "gpt-4o-mini"
        base_url = config.scene_base_url or "https://api.openai.com/v1"
        is_local = any(h in base_url for h in ("localhost", "127.0.0.1", "::1"))
        tools_mode = "prompt" if is_local else "native"
        logger.info("Using OpenAI scene backend: %s @ %s (tools=%s)", model, base_url, tools_mode)
        return OpenAICompatibleBackend(
            api_key=api_key or "local",
            model=model,
            base_url=base_url,
            tools_mode=tools_mode,
        )

    logger.warning("Unknown SCENE_PLATFORM: %s, falling back", platform)
    return None
