"""brief-turn（短い定型ターン）の判定と軽量返信モードのヒューリスティクス（境界R B3b）。

挨拶・相槌・訂正のような短いターンを、高価な想起や探索ツールを避ける軽量経路へ
回すための純関数群。agent.py の staticmethod/classmethod から切り出した（挙動不変）。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..social_policy import SocialPolicyDecision

_BRIEF_GREETING_PATTERNS = (
    r"^おはよ",
    r"^こんにちは",
    r"^こんばんは",
    r"^おーい$",
    r"^もしもし$",
)
_BRIEF_ACK_PATTERNS = (
    r"^ありがとう",
    r"^ありがと",
    r"^助か",
    r"^よかった",
    r"^了解$",
    r"^ok$",
    r"^okay$",
    r"^お願い(?:[。.!！]?信じてる)?$",
    r"^信じてる$",
)
_BRIEF_CORRECTION_PATTERNS = (
    r"言ってない",
    r"勘違",
    r"誤解",
    r"食い違",
    r"そういう意味じゃ",
    r"そうじゃない",
    r"違う",
    r"ちゃう",
    r"^いや[、, ]",
)


def _normalize_brief_turn_text(text: str) -> str:
    """Normalize short conversational turns for lightweight heuristics."""
    return text.strip().lower().rstrip("。.!！?？ ")


def _matches_brief_turn_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    normalized = _normalize_brief_turn_text(text)
    return any(re.search(pattern, normalized) for pattern in patterns)


def is_candidate_brief_turn(user_input: str, *, is_desire_turn: bool) -> bool:
    """Cheap pre-LLM gate for greeting/ack/correction turns.

    These turns should avoid expensive memory recall and exploratory tools.
    """
    if is_desire_turn:
        return False
    text = user_input.strip()
    if not text or len(text) > 80 or "\n" in text:
        return False
    return (
        _matches_brief_turn_pattern(text, _BRIEF_GREETING_PATTERNS)
        or _matches_brief_turn_pattern(text, _BRIEF_ACK_PATTERNS)
        or _matches_brief_turn_pattern(text, _BRIEF_CORRECTION_PATTERNS)
    )


def should_use_brief_reply_mode(
    *,
    user_input: str,
    social_policy: SocialPolicyDecision,
    is_desire_turn: bool,
) -> bool:
    if is_desire_turn:
        return False
    text = user_input.strip()
    if not text or len(text) > 80 or "\n" in text:
        return False
    return social_policy.primary_act in {
        "greeting",
        "acknowledgement",
        "clarification",
        "repair_attempt",
        "boundary_assertion",
        "silence_or_low_presence",
    }


def brief_reply_prompt() -> str:
    return (
        "[Lightweight turn]\n"
        "- This is a short conversational turn.\n"
        "- Reply directly in 1-2 short sentences.\n"
        "- Do not infer plans, facts, or feelings the user did not say.\n"
        "- Do not use observation or memory tools unless explicitly asked."
    )
