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
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

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


# 自己認識 MI を Nudge に含むときの既定重み（Config `self_mi_weight` の pure-function 既定）。
# 旧・activation 上限 C=2.0 の流用をやめ、支配しない薄い錨へ。実行時の値は
# nudge_current_mood が MemoryConfig から注入する。
SELF_KNOWLEDGE_MI_WEIGHT = 0.5

# 自己認識 MI の emotion（PAD）を持つ agent_state キー。既定は中立、REST が書き換える。
SELF_MI_STATE_KEY = "self_mi_emotion"


def compute_n_pad(
    items: "list[tuple[MoodPAD, float]]",
    *,
    self_pad: "MoodPAD | None" = None,
    self_weight: float = SELF_KNOWLEDGE_MI_WEIGHT,
) -> MoodPAD:
    """W の感情トーン N_PAD を activation 加重平均で作る（課題5）。

    `items`＝各 W MI の (PAD, activation 重み)。自己認識 MI（`self_pad`・重み
    `self_weight`）を常に足すので、W が空でも自己 MI emotion を返す（デフォルト感情）。
    `self_pad` 未指定は中立。N_PAD_x=(Σ a_i x_i + C·self_x)/(Σ a_i + C)（C＝self_weight）。
    """
    if self_pad is None:
        self_pad = MoodPAD()
    total_w = self_weight
    sp = self_weight * self_pad.p
    spn = self_weight * self_pad.pn
    sa = self_weight * self_pad.a
    sdom = self_weight * self_pad.dom
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
    行が無ければ中立。mood の更新（nudge・save）は mood-c の `nudge_current_mood`。
    """
    from .db import get_db

    db = get_db()
    with db.lock:
        return load_mood(db.conn())


def _load_mood_with_updated_at(conn) -> "tuple[MoodPAD, datetime | None]":
    """mood と、その最終更新時刻（decay の経過起点）を読む（mood-c）。行が無ければ (中立, None)。"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT value_json, updated_at FROM agent_state WHERE state_key = %s",
            (MOOD_STATE_KEY,),
        )
        row = cur.fetchone()
    if not row:
        return MoodPAD(), None
    updated = row[1]
    if isinstance(updated, str):
        updated = datetime.fromisoformat(updated)
    return MoodPAD.from_json_dict(json.loads(row[0])), updated


def load_self_mi_emotion(conn) -> MoodPAD:
    """自己認識 MI の emotion（PAD）を読む。未設定なら中立（REST が育てる対象）。"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT value_json FROM agent_state WHERE state_key = %s", (SELF_MI_STATE_KEY,)
        )
        row = cur.fetchone()
    if not row:
        return MoodPAD()
    return MoodPAD.from_json_dict(json.loads(row[0]))


def save_self_mi_emotion(conn, pad: MoodPAD) -> None:
    """自己認識 MI の emotion を書き換える（REST 内省が呼ぶ書換口・課題10）。"""
    now = datetime.now(timezone.utc).isoformat()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO agent_state (state_key, value_json, updated_at) VALUES (%s, %s, %s)"
            " ON CONFLICT (state_key) DO UPDATE SET value_json = EXCLUDED.value_json, updated_at = EXCLUDED.updated_at",
            (SELF_MI_STATE_KEY, json.dumps(pad.to_json_dict()), now),
        )


def decay_and_nudge(
    mood: MoodPAD, elapsed_seconds: float, items: "list[tuple[MoodPAD, float]]",
    *,
    self_pad: "MoodPAD | None" = None,
    self_weight: float = SELF_KNOWLEDGE_MI_WEIGHT,
) -> MoodPAD:
    """mood を経過ぶん平静へ減衰させてから W トーン N_PAD へ nudge する（mood-c・純）。

    課題5：`decay_to_rest`（updated_at からの経過）→ `compute_n_pad`（W の PAD の
    activation 加重平均＋自己認識 MI）→ `nudge_toward`。`self_pad`/`self_weight` は
    自己認識 MI の emotion と重み（呼び出し側が store/Config から渡す）。
    """
    decayed = decay_to_rest(mood, elapsed_seconds)
    return nudge_toward(decayed, compute_n_pad(items, self_pad=self_pad, self_weight=self_weight))


def nudge_current_mood(items: "list[tuple[MoodPAD, float]]") -> MoodPAD:
    """現 mood を読み、経過で減衰させ、W トーンで nudge して保存する（mood-c・接続）。

    経過秒＝now − updated_at（行が無ければ0）。評価器の後にターンの W（想起記憶＋現ターン
    感情＋自己認識 MI）から呼ぶ。自己認識 MI の emotion は store から、重みは Config から
    引く。新しい mood を返す。
    """
    from .config import MemoryConfig
    from .db import get_db

    self_weight = MemoryConfig().self_mi_weight
    db = get_db()
    with db.lock:
        conn = db.conn()
        mood, updated_at = _load_mood_with_updated_at(conn)
        self_pad = load_self_mi_emotion(conn)
        if updated_at is not None:
            elapsed = (datetime.now(timezone.utc) - updated_at).total_seconds()
        else:
            elapsed = 0.0
        n_pad = compute_n_pad(items, self_pad=self_pad, self_weight=self_weight)
        new_mood = decay_and_nudge(
            mood, elapsed, items, self_pad=self_pad, self_weight=self_weight
        )
        save_mood(conn, new_mood)
        conn.commit()
    # 観測（#1 感情ループ閉じ・tuning 用）：ターンごとの mood 推移を1行で。debug（本番は切る）。
    logger.debug(
        "MOOD nudge: (%.2f,%.2f,%.2f,%.2f)→(%.2f,%.2f,%.2f,%.2f) "
        "N_PAD=(%.2f,%.2f,%.2f,%.2f) items=%d elapsed=%.0fs",
        mood.p, mood.pn, mood.a, mood.dom,
        new_mood.p, new_mood.pn, new_mood.a, new_mood.dom,
        n_pad.p, n_pad.pn, n_pad.a, n_pad.dom,
        len(items), elapsed,
    )
    return new_mood


def save_mood(conn, mood: MoodPAD) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO agent_state (state_key, value_json, updated_at) VALUES (%s, %s, %s)"
            " ON CONFLICT (state_key) DO UPDATE SET value_json = EXCLUDED.value_json, updated_at = EXCLUDED.updated_at",
            (MOOD_STATE_KEY, json.dumps(mood.to_json_dict()), now),
        )
