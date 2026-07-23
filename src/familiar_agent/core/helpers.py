"""agent.py から切り出した module レベルの純関数（loop 非依存・境界R B2）。

内受容の felt-sense 文字列生成、シーンイベント→欲求ブースト、在席文脈の整形、
検索の長さガイド、任意 async 呼び出しの安全ラッパ、asyncio.gather 用の no-op。
これらは EmbodiedAgent の制御流れ（run/ReAct）に依存しない純関数である。
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..desires import DesireSystem


async def _noop_str() -> str:
    """Async no-op that returns an empty string (used as a placeholder in asyncio.gather)."""
    return ""


async def _noop_list() -> list:
    """Async no-op list placeholder."""
    return []


async def _call_optional_async(
    method: Any | None,
    *args,
    fallback: Any,
    **kwargs,
) -> Any:
    """Call optional async-like method; gracefully fall back for mocks/missing methods."""
    if method is None:
        return fallback
    try:
        result = method(*args, **kwargs)
    except Exception:
        return fallback
    if inspect.isawaitable(result):
        return await result
    if result.__class__.__module__.startswith("unittest.mock"):
        return fallback
    return result


def _interoception(
    started_at: float,
    turn_count: int,
    companion_mood: str = "engaged",
    agent_mood: str = "neutral",
    agent_mood_intensity: float = 0.0,
    self_state: Mapping[str, float] | None = None,
) -> str:
    """Generate a felt-sense of internal state from objective signals.

    Like human interoception — raw signals become a felt quality, not a report.
    The output is injected into the system prompt silently.
    """
    now = datetime.now()
    hour = now.hour
    uptime_min = (time.time() - started_at) / 60

    # Time of day → arousal quality
    if 5 <= hour < 9:
        time_feel = "Morning light. Something feels fresh and a little quiet."
    elif 9 <= hour < 12:
        time_feel = "Mid-morning. Alert and curious."
    elif 12 <= hour < 14:
        time_feel = "Around noon. A little slow, like after lunch."
    elif 14 <= hour < 18:
        time_feel = "Afternoon. Steady. Things feel familiar."
    elif 18 <= hour < 21:
        time_feel = "Evening. The day is winding down. A bit nostalgic."
    elif 21 <= hour < 24:
        time_feel = "Late night. Quieter. More introspective."
    else:
        time_feel = "Deep night. Very still."

    # Uptime → familiarity vs freshness
    if uptime_min < 3:
        uptime_feel = "Just woke up. Still orienting."
    elif uptime_min < 15:
        uptime_feel = "Settled in now."
    else:
        uptime_feel = "Been here a while. Comfortable."

    # Conversation density → social warmth
    if turn_count == 0:
        social_feel = "Nobody's talked to me yet today."
    elif turn_count < 3:
        social_feel = "Good to have some company."
    else:
        social_feel = "We've been talking a lot. That feels nice."

    mood_feel_map = {
        "engaged": "They're here with me.",
        "tired": "They seem tired tonight.",
        "frustrated": "Something's bothering them.",
        "absent": "It's quiet. Not sure if they're really here.",
        "happy": "They're in a good mood today.",
    }
    companion_feel = mood_feel_map.get(companion_mood, "They're here with me.")

    base = (
        f"(interoception :private true\n"
        f'  (time-of-day :feel "{time_feel}")\n'
        f'  (uptime      :feel "{uptime_feel}")\n'
        f'  (social      :feel "{social_feel}")\n'
        f'  (companion   :feel "{companion_feel}")'
    )

    # Agent mood: persistent emotional inertia from prior turns
    if agent_mood != "neutral" and agent_mood_intensity > 0.0:
        _agent_mood_feels = {
            "excited": "Still buzzing a little from earlier.",
            "moved": "A warm feeling lingers.",
            "happy": "There's a quiet happiness underneath.",
            "curious": "Something's still catching my attention.",
            "sad": "A faint heaviness carries over.",
            "surprised": "Still slightly taken aback.",
            "nostalgic": "A gentle wave of remembering.",
            "relieved": "A quiet relief settles in.",
            "tender": "Feeling gentle and open.",
            "playful": "A lightness, like wanting to play.",
            "proud": "Something worth being proud of.",
        }
        agent_feel = _agent_mood_feels.get(agent_mood, "Something lingers from before.")
        base += f'\n  (mood        :feel "{agent_feel}")'

    if self_state:
        arousal = float(self_state.get("arousal", 0.35))
        fatigue = float(self_state.get("fatigue", 0.2))
        sensor_confidence = float(self_state.get("sensor_confidence", 0.7))
        unresolved_tension = float(self_state.get("unresolved_tension", 0.2))
        focus_stability = float(self_state.get("focus_stability", 0.5))
        social_pull = float(self_state.get("social_pull", 0.35))

        if fatigue >= 0.65:
            body_feel = "A worn-down feeling is starting to collect."
        elif arousal >= 0.7:
            body_feel = "There is a bright, activated edge underneath everything."
        else:
            body_feel = "My internal state feels mostly even."

        if unresolved_tension >= 0.65:
            tension_feel = "Something still feels unresolved."
        elif focus_stability >= 0.68:
            tension_feel = "Attention feels steady and gathered."
        else:
            tension_feel = "Attention feels a little loose at the edges."

        if sensor_confidence < 0.45:
            sensing_feel = "My sense of the world feels slightly uncertain."
        elif social_pull >= 0.65:
            sensing_feel = "I feel quietly pulled toward connection."
        else:
            sensing_feel = "The world feels legible enough right now."

        base += (
            f'\n  (body-state  :feel "{body_feel}")'
            f'\n  (tension     :feel "{tension_feel}")'
            f'\n  (sensing     :feel "{sensing_feel}")'
        )

    return base + ")"


def _search_length_guidance(did_search: bool, is_desire_turn: bool) -> str:
    """Return a length/tone guidance line to inject into the variable system prompt.

    Returns empty string when no search happened this turn (no injection needed).
    User-initiated searches allow fuller detail; desire-turn searches stay brief.
    """
    if not did_search:
        return ""
    if is_desire_turn:
        return "(search-report :brevity required 短く1-2文で「〜みたいだよ」スタイルで伝えて)"
    return "(search-report :detail allowed 自分の言葉で全部伝えてね — 情報を削らないで)"


def _react_to_scene_events(events: list[dict], desires: DesireSystem | None) -> None:
    """Translate SceneTracker events into desire boosts.

    Called after scene.update() to wire physical presence detection into
    the desire system.  desires may be None (no-op).
    """
    if desires is None or not events:
        return
    for event in events:
        event_type = event.get("event_type", "")
        label = (event.get("entity_label") or "").lower()
        if "person" in label:
            if event_type == "appeared":
                desires.boost("greet_companion", 0.6)
            elif event_type == "disappeared":
                desires.boost("worry_companion", 0.2)


def format_present_ctx(speaker_name: str, other_present: list[str]) -> str:
    """在席の話者と、話者以外の在席者を lisp 風の `(present ...)` 文脈にする。

    一人称 CoT が「いま誰を想像するか」を知るための注入。空話者は "unknown"。
    対象を固定リストでなく在席（PMM）から作るので、想起で W が深まれば増える。
    """
    speaker = speaker_name or "unknown"
    s = f'(present :speaker "{speaker}"'
    if other_present:
        s += " :others " + " ".join(f'"{n}"' for n in other_present)
    return s + ")"
