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


# 自己認識 MI のフラット emotion(0.5×4) を Nudge に含むときの重み（課題5 の C＝
# activation 上限）。pinned なので常に効き、W が薄いと mood は中立へ寄る。Config 差し替え可。
SELF_KNOWLEDGE_MI_WEIGHT = 2.0


def compute_n_pad(
    items: "list[tuple[MoodPAD, float]]",
    *,
    self_weight: float = SELF_KNOWLEDGE_MI_WEIGHT,
) -> MoodPAD:
    """W の感情トーン N_PAD を activation 加重平均で作る（課題5・mood-a・未接続）。

    `items`＝各 W MI の (PAD, activation 重み)。自己認識 MI のフラット項
    (0.5,0.5,0.5,0.5)・重み `self_weight` を常に足すので、W が空でも中立を返す。
    N_PAD_x =(Σ a_i x_i + C·0.5)/(Σ a_i + C)（x＝p,pn,a,dom・C＝self_weight）。
    """
    total_w = self_weight
    sp = self_weight * 0.5
    spn = self_weight * 0.5
    sa = self_weight * 0.5
    sdom = self_weight * 0.5
    for pad, w in items:
        total_w += w
        sp += w * pad.p
        spn += w * pad.pn
        sa += w * pad.a
        sdom += w * pad.dom
    return MoodPAD(sp / total_w, spn / total_w, sa / total_w, sdom / total_w).clipped()


def nudge_toward(mood: MoodPAD, n_pad: MoodPAD) -> MoodPAD:
    """W トーン N_PAD で mood を動かす（課題5・mood-a・未接続）。

    覚醒が高いほど強く引かれる：A_M←max(A_M,A_N)／X_M←X_M+A_N(X_N−X_M)（X＝p,pn,dom）。
    push でなく漸近なので Dom の意味も壊れない。接続は mood-c。
    """
    a_n = n_pad.a
    return MoodPAD(
        p=mood.p + a_n * (n_pad.p - mood.p),
        pn=mood.pn + a_n * (n_pad.pn - mood.pn),
        a=max(mood.a, a_n),
        dom=mood.dom + a_n * (n_pad.dom - mood.dom),
    ).clipped()


def load_current_mood() -> MoodPAD:
    """自己接続で現在の mood を読む（W2b-2・読みだけ）。

    他状態モジュール（self_state 等）と同じく get_db() で接続し `load_mood` を呼ぶ。
    行が無ければ中立。mood の更新（nudge・save）は後段の mood スライスで繋ぐ。
    """
    from .db import get_db

    db = get_db()
    with db.lock:
        return load_mood(db.conn())


def save_mood(conn, mood: MoodPAD) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO agent_state (state_key, value_json, updated_at) VALUES (%s, %s, %s)"
            " ON CONFLICT (state_key) DO UPDATE SET value_json = EXCLUDED.value_json, updated_at = EXCLUDED.updated_at",
            (MOOD_STATE_KEY, json.dumps(mood.to_json_dict()), now),
        )
