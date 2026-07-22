"""New-5 drive register (SEEKING, REST, BOND, SAFETY, ESTEEM).

Holds the new 5-drive vector `AiDrivers`, each axis in [0,1] with a still
(default) value of 0.0, persisted under agent_state key "drive5".

This is a thin vertical slice (Phase 1 B-2): the register and its
persistence exist here but are not wired to anything — not to the live
15-drive `DesireSystem` (desires.py, state_key "desires"), not to
`as_coalition`, not to agent.py's desire usage. Accumulation, discharge,
and mood modulation (dynamics) are later work, once the mood register
(B-1) is connected.

`AiDrivers` is distinct from mental_state.py's `DriveVector` — that module
is unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone

DRIVE_STATE_KEY = "drive5"


@dataclass(frozen=True)
class AiDrivers:
    seeking: float = 0.0
    rest: float = 0.0
    bond: float = 0.0
    safety: float = 0.0
    esteem: float = 0.0

    def clipped(self) -> "AiDrivers":
        return replace(
            self,
            seeking=max(0.0, min(1.0, self.seeking)),
            rest=max(0.0, min(1.0, self.rest)),
            bond=max(0.0, min(1.0, self.bond)),
            safety=max(0.0, min(1.0, self.safety)),
            esteem=max(0.0, min(1.0, self.esteem)),
        )

    def to_json_dict(self) -> dict:
        return {
            "seeking": self.seeking, "rest": self.rest, "bond": self.bond,
            "safety": self.safety, "esteem": self.esteem,
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "AiDrivers":
        return cls(
            seeking=data.get("seeking", 0.0),
            rest=data.get("rest", 0.0),
            bond=data.get("bond", 0.0),
            safety=data.get("safety", 0.0),
            esteem=data.get("esteem", 0.0),
        )


def load_drives(conn) -> AiDrivers:
    with conn.cursor() as cur:
        cur.execute("SELECT value_json FROM agent_state WHERE state_key = %s", (DRIVE_STATE_KEY,))
        row = cur.fetchone()
    if not row:
        return AiDrivers()
    return AiDrivers.from_json_dict(json.loads(row[0]))


def load_current_drives() -> AiDrivers:
    """自己接続で現在の drive5 を読む（読みだけ・`load_current_mood` と同型）。

    GUI など、接続を持たない読み取り側のために get_db() で繋いで `load_drives` を呼ぶ。
    行が無ければ既定（全軸 0.0）。
    """
    from .db import get_db

    db = get_db()
    with db.lock:
        return load_drives(db.conn())


def save_drives(conn, drives: AiDrivers) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO agent_state (state_key, value_json, updated_at) VALUES (%s, %s, %s)"
            " ON CONFLICT (state_key) DO UPDATE SET value_json = EXCLUDED.value_json, updated_at = EXCLUDED.updated_at",
            (DRIVE_STATE_KEY, json.dumps(drives.to_json_dict()), now),
        )
