"""Shared UI utilities for TUI, GUI, and REPL.

This module is the single source of truth for:
  - ACTION_ICONS: icon mapping for tool calls
  - format_action(): human-readable tool-call label

Keeping these here prevents duplication across tui.py, gui.py, and main.py.
"""

from __future__ import annotations

import re
from datetime import datetime

from ._i18n import _t


# ---------------------------------------------------------------------------
# Chat log formatting
# ---------------------------------------------------------------------------


def format_chat_log_line(text: str, now: datetime | None = None) -> str:
    """Return text with a timestamp prepended for chat.log file writes.

    File-write only — do not use for screen bubbles. Format matches app.log
    so timestamps can be cross-referenced line by line.
    """
    ts = (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    return f"[{ts}] {text}"


# ---------------------------------------------------------------------------
# Spoken-text cleaning
# ---------------------------------------------------------------------------

# TTS audio-direction tags, e.g. [cheerful] [laughs] [whispers].
# ElevenLabs eleven_v3 interprets these natively as audio tags, so they must
# survive on the TTS path even though they are stripped from visual display.
_TTS_TAG_RE = re.compile(r"\[[^\]]*\]")
# Stage directions in parentheses, full-width （…） or half-width (…).
# ElevenLabs does NOT treat these as audio tags — it would read them aloud —
# and they are visual narration, so they are dropped from both voice and display.
_STAGE_DIRECTION_RE = re.compile(r"（[^）]*）|\([^)]*\)")


def _collapse_ws(text: str) -> str:
    """Collapse runs of spaces (incl. full-width) left behind by removals."""
    return re.sub(r"[ 　\t]{2,}", " ", text).strip()


def strip_stage_directions(text: str) -> str:
    """Remove （…）/(…) parenthetical narration, preserving [audio tags].

    Used on the TTS path: parenthetical asides like '（静かに待つ）' have no
    spoken meaning and ElevenLabs would read them literally, but [whispers]-style
    audio tags are native eleven_v3 directives and must reach the API intact.
    """
    return _collapse_ws(_STAGE_DIRECTION_RE.sub("", text))


def clean_spoken_text(text: str) -> str:
    """Strip BOTH [audio tags] and （…）/(…) stage directions for visual display.

    Removes [bracket-tag] audio codes and parenthetical narration, then collapses
    leftover whitespace. Use for the chat log / display, where neither should be
    shown. For the TTS path use ``strip_stage_directions`` instead so audio tags
    survive.
    """
    return _collapse_ws(_STAGE_DIRECTION_RE.sub("", _TTS_TAG_RE.sub("", text)))


# ---------------------------------------------------------------------------
# Action icons (single source of truth)
# ---------------------------------------------------------------------------

ACTION_ICONS: dict[str, str] = {
    "see": "👀",
    "look": "🔄",
    "look_left": "◀️",
    "look_right": "▶️",
    "look_up": "🔼",
    "look_down": "🔽",
    "look_around": "🔄",
    "walk": "🚶",
    "say": "🗣️",
    "remember": "💾",
    "recall": "💭",
    "listen": "🎙️",
    "search": "🔍",
    "brave_web_search": "🔍",
    "brave_local_search": "📍",
}

# Tool names that have dedicated i18n labels (key: "action_{name}")
_I18N_ACTION_NAMES: frozenset[str] = frozenset({"see", "look", "walk", "say", "remember", "recall"})


def format_action(name: str, tool_input: dict) -> str:
    """Return a human-readable label for a tool call.

    Used identically by TUI, GUI, and REPL to display tool invocations.
    """
    icon = ACTION_ICONS.get(name, "⚙")

    if name == "look":
        direction = tool_input.get("direction", "")
        key = {
            "left": "look_left",
            "right": "look_right",
            "up": "look_up",
            "down": "look_down",
        }.get(direction, "look_around")
        dir_icon = ACTION_ICONS.get(key, icon)
        deg = tool_input.get("degrees", "")
        suffix = f"({deg}°)" if deg else ""
        return f"{dir_icon} {_t(key)}{suffix}"

    if name == "walk":
        direction = tool_input.get("direction", "?")
        duration = tool_input.get("duration")
        if duration:
            return f"{icon} {_t('walk_timed', direction=direction, duration=str(duration))}"
        return f"{icon} {_t('walk_dir', direction=direction)}"

    if name == "say":
        raw = str(tool_input.get("text", ""))
        preview = raw[:50]
        ellipsis = "…" if len(raw) > 50 else ""
        return f"{icon} 「{preview}{ellipsis}」"

    if name in _I18N_ACTION_NAMES:
        try:
            return _t(f"action_{name}")
        except KeyError:
            pass

    return f"{icon} {name}"


# ---------------------------------------------------------------------------
# Tool result formatting (shown AFTER a tool runs, in a result bubble)
# ---------------------------------------------------------------------------

# Tools whose results we want to surface in the UI

_EMOTION_COLORS: dict[str, str] = {
    "happy": "🌸",
    "sad": "💧",
    "curious": "✨",
    "excited": "⚡",
    "moved": "💫",
    "neutral": "·",
}


# ---------------------------------------------------------------------------
# Idle wait (UI-agnostic core)
# ---------------------------------------------------------------------------

IDLE_CHECK_INTERVAL: float = 10.0  # 入力待ちのタイムアウト（秒）。GUI・TUI・REPL の入力ループが使う
