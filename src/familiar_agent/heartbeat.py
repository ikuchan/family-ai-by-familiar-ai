"""Heartbeat runtime for continuation control and routine-time behavior."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import psycopg2.extras

from .db import get_db
from .routines import QuietHoursRule, RoutineDecision, evaluate_routine_state, load_optional_notes

if TYPE_CHECKING:
    from .tools.memory import ObservationMemory

logger = logging.getLogger(__name__)

DONE = "DONE"


@dataclass(slots=True, frozen=True)
class ContinuationDecision:
    status: str
    chain_depth: int
    persisted_remainder: bool = False


@dataclass(slots=True)
class HeartbeatState:
    chain_depth: int = 0
    last_status: str = DONE
    last_continuation_reason: str = ""
    last_updated_at: str = ""


class HeartbeatRuntime:
    """Session heartbeat for routine turns and bounded continuation."""

    def __init__(
        self,
        *,
        memory: ObservationMemory | None = None,
        quiet_rule: QuietHoursRule | None = None,
        base_dir: Path | None = None,
        state_path: Path | None = None,  # ignored; kept for call-site compatibility
        max_chain_depth: int = 3,
    ) -> None:
        self._memory = memory
        self._quiet_rule = quiet_rule or QuietHoursRule()
        self._base_dir = base_dir or Path.cwd()
        self._max_chain_depth = max_chain_depth
        self._state = self._load_state()
        self._chain_depth = self._state.chain_depth
        self._last_continuation_reason = self._state.last_continuation_reason

    def _load_state(self) -> HeartbeatState:
        try:
            db = get_db()
            with db.lock:
                conn = db.conn()
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("SELECT value_json FROM agent_state WHERE state_key = %s", ("heartbeat",))
                    row = cur.fetchone()
            if row:
                payload = json.loads(row["value_json"])
                if isinstance(payload, dict):
                    return HeartbeatState(
                        chain_depth=max(0, int(payload.get("chain_depth", 0))),
                        last_status=str(payload.get("last_status", DONE)),
                        last_continuation_reason=str(payload.get("last_continuation_reason", "")),
                        last_updated_at=str(payload.get("last_updated_at", "")),
                    )
        except Exception as exc:
            logger.warning("Could not load heartbeat state: %s", exc)
        return HeartbeatState()

    def _save_state(self, status: str) -> None:
        self._state = HeartbeatState(
            chain_depth=self._chain_depth,
            last_status=status,
            last_continuation_reason=self._last_continuation_reason,
            last_updated_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            now = self._state.last_updated_at
            db = get_db()
            with db.lock:
                conn = db.conn()
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO agent_state (state_key, value_json, updated_at) VALUES (%s, %s, %s)"
                        " ON CONFLICT (state_key) DO UPDATE SET value_json = EXCLUDED.value_json, updated_at = EXCLUDED.updated_at",
                        ("heartbeat", json.dumps(asdict(self._state), ensure_ascii=False), now),
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("Could not save heartbeat state: %s", exc)

    def routine_state(self, now: datetime | None = None) -> RoutineDecision:
        return evaluate_routine_state(self._quiet_rule, now)

    def morning_reconstruction_notes(self) -> str:
        notes = load_optional_notes(self._base_dir)
        lines = ["[Routine notes]"]
        continuation = self.continuity_context_for_prompt()
        if continuation:
            lines.append(f"- carryover: {continuation}")
        for name, text in notes.items():
            if text:
                lines.append(f"- {name}: {text[:180]}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def continuity_context_for_prompt(self) -> str:
        if not self._last_continuation_reason:
            return ""
        status = self._state.last_status
        if status.startswith("DEFER:"):
            return f"deferred-thread={self._last_continuation_reason[:120]}"
        if status.startswith("CONTINUE:"):
            return (
                f"active-continuation={self._last_continuation_reason[:120]} "
                f"(depth={self._chain_depth})"
            )
        return ""

    def parse_status(self, text: str) -> str:
        stripped = (text or "").strip()
        if not stripped:
            return DONE
        if stripped.startswith("CONTINUE:"):
            return stripped
        if stripped.startswith("DEFER:"):
            return stripped
        if stripped == DONE:
            return DONE
        return DONE

    def apply_status(self, status: str) -> ContinuationDecision:
        parsed = self.parse_status(status)
        if parsed == DONE:
            self._chain_depth = 0
            self._last_continuation_reason = ""
            self._save_state(DONE)
            return ContinuationDecision(status=DONE, chain_depth=0, persisted_remainder=False)

        if parsed.startswith("DEFER:"):
            reason = parsed.split(":", 1)[1].strip() or "deferred"
            self._persist_remainder(reason)
            self._chain_depth = 0
            self._last_continuation_reason = reason
            self._save_state(parsed)
            return ContinuationDecision(status=parsed, chain_depth=0, persisted_remainder=True)

        reason = parsed.split(":", 1)[1].strip() if ":" in parsed else "continued"
        self._chain_depth += 1
        self._last_continuation_reason = reason
        if self._chain_depth > self._max_chain_depth:
            self._persist_remainder(reason)
            self._chain_depth = 0
            self._last_continuation_reason = reason
            self._save_state(f"DEFER:{reason}")
            return ContinuationDecision(
                status=f"DEFER:{reason}",
                chain_depth=0,
                persisted_remainder=True,
            )
        self._save_state(parsed)
        return ContinuationDecision(status=parsed, chain_depth=self._chain_depth)

    def _persist_remainder(self, reason: str) -> None:
        if self._memory is None:
            return
        try:
            self._memory.open_unfinished_business(
                summary=reason,
                source="continuation",
                metadata={"status": "deferred"},
            )
        except Exception:
            return
