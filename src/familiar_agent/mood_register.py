"""PAD (Pleasure/Pain, Arousal, Dominance) mood register.

Holds mood M as four independent axes (p, pn, a, dom), each decaying
toward the midpoint rest state M_rest=(0.5,0.5,0.5,0.5) with a half-life
of HALF_LIFE_SECONDS=600s, persisted under agent_state key "mood_pad".

Unlike time_decay.DecayState (which decays a value toward a floor), mood
axes below 0.5 rise back toward it, so this module decays toward the
midpoint rather than toward zero.

This is a thin vertical slice (Phase 1 B-1): the register, its decay
function, and its persistence exist here but are not wired to anything —
not to agent.py's `_mood`/`_decayed_mood`, not to mental_state.py's
AffectiveState, and not to the emotion->PAD mapping phi (deferred to
issue #11k). Connecting it to those is later work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone

REST = 0.5
HALF_LIFE_SECONDS = 600.0
MOOD_STATE_KEY = "mood_pad"


@dataclass(frozen=True)
class MoodPAD:
    p: float = 0.5
    pn: float = 0.5
    a: float = 0.5
    dom: float = 0.5

    def clipped(self) -> "MoodPAD":
        return replace(
            self,
            p=max(0.0, min(1.0, self.p)),
            pn=max(0.0, min(1.0, self.pn)),
            a=max(0.0, min(1.0, self.a)),
            dom=max(0.0, min(1.0, self.dom)),
        )

    def to_json_dict(self) -> dict:
        return {"p": self.p, "pn": self.pn, "a": self.a, "dom": self.dom}

    @classmethod
    def from_json_dict(cls, data: dict) -> "MoodPAD":
        return cls(
            p=data.get("p", REST),
            pn=data.get("pn", REST),
            a=data.get("a", REST),
            dom=data.get("dom", REST),
        )


def decay_to_rest(
    mood: MoodPAD, elapsed_seconds: float, *, rest: float = REST, half_life: float = HALF_LIFE_SECONDS,
) -> MoodPAD:
    """Decay each axis toward `rest`, halving the distance every `half_life` seconds."""
    factor = 2.0 ** (-max(0.0, elapsed_seconds) / half_life)
    return MoodPAD(
        p=rest + (mood.p - rest) * factor,
        pn=rest + (mood.pn - rest) * factor,
        a=rest + (mood.a - rest) * factor,
        dom=rest + (mood.dom - rest) * factor,
    ).clipped()


def load_mood(conn) -> MoodPAD:
    with conn.cursor() as cur:
        cur.execute("SELECT value_json FROM agent_state WHERE state_key = %s", (MOOD_STATE_KEY,))
        row = cur.fetchone()
    if not row:
        return MoodPAD()
    return MoodPAD.from_json_dict(json.loads(row[0]))


def save_mood(conn, mood: MoodPAD) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO agent_state (state_key, value_json, updated_at) VALUES (%s, %s, %s)"
            " ON CONFLICT (state_key) DO UPDATE SET value_json = EXCLUDED.value_json, updated_at = EXCLUDED.updated_at",
            (MOOD_STATE_KEY, json.dumps(mood.to_json_dict()), now),
        )
