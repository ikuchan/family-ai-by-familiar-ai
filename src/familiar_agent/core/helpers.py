"""agent.py から切り出した module レベルの純関数（loop 非依存・境界R B2）。

内受容の felt-sense 文字列生成、シーンイベント→欲求ブースト、在席文脈の整形、
検索の長さガイド、任意 async 呼び出しの安全ラッパ、asyncio.gather 用の no-op。
これらは EmbodiedAgent の制御流れ（run/ReAct）に依存しない純関数である。
"""

from __future__ import annotations

import inspect
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
