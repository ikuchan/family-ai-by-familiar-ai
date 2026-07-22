"""Shared UI utilities for TUI, GUI, and REPL.

This module is the single source of truth for:
  - ACTION_ICONS: icon mapping for tool calls
  - format_action(): human-readable tool-call label
  - should_fire_idle_desire(): shared gate for autonomous desire turns
  - desire_tick_prompt(): extract the current dominant desire prompt (UI-agnostic)

Keeping these here prevents duplication across tui.py, gui.py, and main.py.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import TYPE_CHECKING

from ._i18n import _t

if TYPE_CHECKING:
    from .desires import DesireSystem


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
_RESULT_DISPLAY_TOOLS: frozenset[str] = frozenset({"remember", "recall"})

_EMOTION_COLORS: dict[str, str] = {
    "happy": "🌸",
    "sad": "💧",
    "curious": "✨",
    "excited": "⚡",
    "moved": "💫",
    "neutral": "·",
}


def format_tool_result(name: str, _tool_input: dict, result: str) -> str | None:
    """Return a formatted string to display after a tool runs, or None to suppress.

    Called by GUI/TUI after on_action (which fires before the tool runs).
    Only memory/recall results are surfaced; other tools are silent.
    """
    if name not in _RESULT_DISPLAY_TOOLS:
        return None

    if name == "remember":
        return _format_remember_result(result)
    if name == "recall":
        return _format_recall_result(result)
    return None


def _format_remember_result(result: str) -> str:
    """Format remember() result into a terse confirmation line."""
    # result looks like: "Remembered [id:XXXX]: content\nemotion=X | id=FULL_UUID"
    lines = result.splitlines()
    if not lines:
        return result
    first = lines[0]
    meta = lines[1] if len(lines) > 1 else ""

    emotion = "neutral"
    if "emotion=" in meta:
        emotion = meta.split("emotion=")[1].split("|")[0].strip()

    emo_icon = _EMOTION_COLORS.get(emotion, "·")
    # Extract content preview (strip the "Remembered [id:XXXX]: " prefix)
    content = first
    if ": " in first:
        content = first.split(": ", 1)[1].strip()

    # Extract short id
    short_id = ""
    if "id:" in first:
        part = first.split("id:")[1].split("]")[0][:8]
        short_id = f" #{part}"

    return f"  {emo_icon} {content[:80]}{short_id}"


def _format_recall_result(result: str) -> str:
    """Format recall() result — keep it compact but readable."""
    if result == "No relevant memories found.":
        return "  · (記憶なし)"

    lines = result.splitlines()
    out: list[str] = []
    item_count = sum(1 for ln in lines if ln.startswith("- "))
    out.append(f"  {item_count}件 ↩")

    for line in lines:
        if line.startswith("- ["):
            # Parse "- [emotion] date time id:XXXX src:kind ..."
            # then the next line has the content
            parts = line[3:].split("]", 1)
            emo = parts[0] if parts else "neutral"
            rest = parts[1].strip() if len(parts) > 1 else ""
            # Extract date
            date_part = rest.split(" ")[0] if rest else ""
            emo_icon = _EMOTION_COLORS.get(emo, "·")
            out.append(f"  {emo_icon} [{emo}] {date_part}")
        elif line.startswith("  ") and not line.startswith("  →") and not line.startswith("  ←"):
            # Content line
            out.append(f"    {line.strip()[:100]}")
        elif line.startswith("  →") or line.startswith("  ←"):
            # Linked memory
            out.append(f"    {line.strip()[:90]}")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Desire tick (UI-agnostic core)
# ---------------------------------------------------------------------------

IDLE_CHECK_INTERVAL: float = 10.0  # seconds between desire checks when idle
DESIRE_COOLDOWN: float = float(os.environ.get("DESIRE_COOLDOWN", "90"))  # configurable
INTERNAL_DESIRE_COOLDOWN: float = float(os.environ.get("INTERNAL_DESIRE_COOLDOWN", str(DESIRE_COOLDOWN)))

# Adaptive cooldown settings for social desires
SOCIAL_DESIRE_COOLDOWN_BASE: float = float(os.environ.get("SOCIAL_DESIRE_COOLDOWN", "30"))
SOCIAL_DESIRE_COOLDOWN_MAX: float = float(os.environ.get("SOCIAL_DESIRE_COOLDOWN_MAX", "120"))

# How long to stay silent after a silence request (seconds)
SILENCE_DURATION_SEC: float = float(os.environ.get("SILENCE_DURATION", "1800"))

_SILENCE_KEYWORDS: tuple[str, ...] = (
    "静かにして", "しずかにして", "黙って", "うるさい",
    "話しかけないで", "今は話しかけないで", "放っておいて", "ほっておいて",
    "be quiet", "stop talking", "leave me alone", "don't talk to me", "shut up",
)


def is_silence_request(text: str) -> bool:
    """Return True if the user is asking the agent to stop talking."""
    lower = text.lower()
    return any(kw in lower for kw in _SILENCE_KEYWORDS)


class AdaptiveDesireCooldown:
    """Adaptive cooldown for social desires.

    Sequence: base → base×2 → base×4 (capped at max).
    Resets to base when the user sends any message.
    Step advances only when a social desire fires without a prior user response.
    """

    _MULTIPLIERS: tuple[int, ...] = (1, 2, 4)

    def __init__(
        self,
        base: float = SOCIAL_DESIRE_COOLDOWN_BASE,
        max_cd: float = SOCIAL_DESIRE_COOLDOWN_MAX,
    ) -> None:
        self._base = base
        self._max = max_cd
        self._step = 0
        self._user_responded = True  # treat startup as "user just spoke"

    @property
    def current(self) -> float:
        mult = self._MULTIPLIERS[min(self._step, len(self._MULTIPLIERS) - 1)]
        return min(self._max, self._base * mult)

    def on_desire_fired(self) -> None:
        """Call after a social desire fires."""
        if not self._user_responded:
            self._step = min(self._step + 1, len(self._MULTIPLIERS) - 1)
        self._user_responded = False

    def on_user_message(self) -> None:
        """Call when the user sends any message — resets to base cooldown."""
        self._step = 0
        self._user_responded = True


def should_fire_idle_desire(
    *,
    agent_running: bool,
    has_pending_input: bool,
    last_interaction: float,
    now: float,
    cooldown: float = DESIRE_COOLDOWN,
) -> bool:
    """Return True when an autonomous desire turn is allowed to fire."""
    if agent_running:
        return False
    if has_pending_input:
        return False
    return now - last_interaction >= cooldown


def desire_tick_prompt(
    desires: DesireSystem,
    input_queue_peek: list[str],
) -> tuple[str, str, str | None] | None:
    """Return (desire_name, prompt, pending_note) or None if no desire fires.

    Args:
        desires: the DesireSystem to query.
        input_queue_peek: list of items currently in the input queue
            (caller drains it and passes the contents here so this
            function can fold any pending user note into the prompt
            without touching the queue itself).

    Returns:
        (desire_name, prompt, pending_note) if a desire is ready to fire.
        None if no dominant desire exists or prompt is empty.
    """
    prompt = desires.dominant_as_prompt()
    if not prompt:
        return None

    dominant = desires.get_dominant()
    if dominant is None:
        return None

    desire_name, _ = dominant

    pending_note: str | None = None
    if input_queue_peek:
        pending_note = input_queue_peek[0]
        prompt = _t("desire_pending_note", note=pending_note, prompt=prompt)

    return desire_name, prompt, pending_note
