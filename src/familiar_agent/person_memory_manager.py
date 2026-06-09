"""Person-aware memory routing layer.

Architecture
------------
present_persons : dict[person_id, PersonPresence]
    Everyone currently in the room.
current_speaker_id : str | None
    The person actively speaking this turn.

Memory access
-------------
- Writes  → current_speaker's memory space.
- Reads   → situated search across ALL present persons + AGENT_SELF.
- Agent's own memories (self_model, curiosity …) always use AGENT_SELF_ID.

Perspective vectors (design α)
-------------------------------
Each person has a perspective_vec stored in persons.perspective_vec.
At memory-write time every registered person gets a situated_embedding
pre-computed as:  normalise(mem_vec + ALPHA * person_vec).
At recall time the query is issued directly against situated_embeddings,
returning sorted results from SQL — no full table scan needed.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable

import numpy as np

logger = logging.getLogger(__name__)

# ── Reserved person IDs ────────────────────────────────────────────────────
AGENT_SELF_ID     = "00000000-0000-0000-0000-000000000000"
DEFAULT_PERSON_ID = "00000000-0000-0000-0000-000000000001"

# Perspective vector blend weight (0 = no perspective, 1 = full person bias)
ALPHA: float = 0.30
# Auto-switch threshold for non-manual/llm hints
AUTO_SWITCH_THRESHOLD: float = 0.75
# Seconds without camera/voice signal before requesting re-detection
PRESENCE_TIMEOUT_SEC: float = 120.0


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class PersonPresence:
    person_id: str
    confidence: float = 1.0
    arrived_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_signal_at: float = field(default_factory=time.time)


@dataclass
class RecognitionHint:
    """A single recognition signal from any source."""
    person_id: str
    confidence: float          # 0.0–1.0
    source: str                # "face" | "voice" | "text" | "manual" | "llm"
    reason: str = ""

    # Special source values that bypass the confidence threshold
    IMMEDIATE = frozenset({"manual", "llm"})


# ── Manager ────────────────────────────────────────────────────────────────

class PersonMemoryManager:
    """Central coordinator for person identity and memory routing."""

    def __init__(self, base_memory: "ObservationMemory") -> None:  # type: ignore[name-defined]
        self._base = base_memory
        self._present: dict[str, PersonPresence] = {}
        self._speaker_id: str | None = None
        self._instances: dict[str, Any] = {}   # person_id → ObservationMemory
        self._switch_callbacks: list[Callable[[str | None, str], Awaitable[None]]] = []
        self._lock = threading.Lock()

    # ── Presence management ────────────────────────────────────────────────

    async def person_arrived(self, person_id: str, confidence: float = 1.0) -> None:
        """Register that someone has entered the space."""
        with self._lock:
            was_empty = len(self._present) == 0
            self._present[person_id] = PersonPresence(
                person_id=person_id, confidence=confidence
            )
        logger.info("Arrived: %s  (total present: %d)", person_id, len(self._present))
        if was_empty:
            await self.set_speaker(person_id, source="auto")

    async def person_left(self, person_id: str) -> None:
        """Register that someone has left the space."""
        with self._lock:
            self._present.pop(person_id, None)
            if self._speaker_id == person_id:
                self._speaker_id = None
        logger.info("Left: %s  (total present: %d)", person_id, len(self._present))

    def get_present_ids(self) -> list[str]:
        with self._lock:
            return list(self._present.keys())

    def refresh_signal(self, person_id: str) -> None:
        """Update the last-signal timestamp for a present person."""
        with self._lock:
            if person_id in self._present:
                self._present[person_id].last_signal_at = time.time()

    def stale_present_ids(self) -> list[str]:
        """Return IDs of persons whose last signal is older than PRESENCE_TIMEOUT_SEC."""
        now = time.time()
        with self._lock:
            return [
                pid for pid, p in self._present.items()
                if (now - p.last_signal_at) > PRESENCE_TIMEOUT_SEC
            ]

    # ── Speaker management ─────────────────────────────────────────────────

    async def set_speaker(self, person_id: str, source: str = "manual") -> bool:
        """Declare who is speaking. Adds them to present if not already."""
        if person_id not in self._present:
            await self.person_arrived(person_id)
        old = self._speaker_id
        with self._lock:
            self._speaker_id = person_id
        if old != person_id:
            logger.info("Speaker: %s → %s  (source=%s)", old, person_id, source)
            for cb in self._switch_callbacks:
                try:
                    await cb(old, person_id)
                except Exception as e:
                    logger.warning("Switch callback error: %s", e)
        return old != person_id

    @property
    def current_speaker_id(self) -> str | None:
        return self._speaker_id

    def on_switch(self, callback: Callable[[str | None, str], Awaitable[None]]) -> None:
        self._switch_callbacks.append(callback)

    # ── Recognition hint processing ────────────────────────────────────────

    async def apply_hint(self, hint: RecognitionHint) -> bool:
        """Apply a recognition signal. Returns True if a switch happened."""
        self.refresh_signal(hint.person_id)
        if hint.source in hint.IMMEDIATE:
            return await self.set_speaker(hint.person_id, source=hint.source)
        if hint.confidence >= AUTO_SWITCH_THRESHOLD:
            return await self.set_speaker(hint.person_id, source=hint.source)
        logger.debug(
            "Hint below threshold: src=%s pid=%s conf=%.2f",
            hint.source, hint.person_id[:8], hint.confidence,
        )
        return False

    # ── Memory access ──────────────────────────────────────────────────────

    def get_speaker_memory(self) -> "ObservationMemory | None":  # type: ignore[name-defined]
        """Write target: current speaker's memory. None if no speaker known."""
        if self._speaker_id is None:
            return None
        return self._get_or_create(self._speaker_id)

    def get_memory_for(self, person_id: str) -> "ObservationMemory":  # type: ignore[name-defined]
        return self._get_or_create(person_id)

    def get_agent_memory(self) -> "ObservationMemory":  # type: ignore[name-defined]
        return self._get_or_create(AGENT_SELF_ID)

    def get_all_present_memories(
        self,
    ) -> list[tuple[str, "ObservationMemory"]]:  # type: ignore[name-defined]
        """Read targets: memories of all present persons."""
        return [(pid, self._get_or_create(pid)) for pid in self.get_present_ids()]

    def _get_or_create(self, person_id: str) -> "ObservationMemory":  # type: ignore[name-defined]
        if person_id not in self._instances:
            self._instances[person_id] = self._base.for_person(person_id)
        return self._instances[person_id]

    # ── Person registry helpers ────────────────────────────────────────────

    def register_person(self, name: str, display_name: str = "") -> str:
        return self._base.register_person(name, display_name)

    def list_persons(self) -> list[dict]:
        return self._base.list_persons()

    def get_speaker_info(self) -> dict | None:
        if self._speaker_id is None:
            return None
        persons = {p["id"]: p for p in self.list_persons()}
        return persons.get(self._speaker_id)

    def get_person_name(self, person_id: str) -> str:
        persons = {p["id"]: p for p in self.list_persons()}
        return persons.get(person_id, {}).get("display_name", person_id[:8])

    def find_person_id_by_name(self, name: str) -> str | None:
        """Look up person UUID by name field or any alias in display_name.

        display_name may contain comma/読点-separated aliases such as
        "パパ、いくながさん、ゆうすけ".  Checks each alias individually.
        """
        for p in self.list_persons():
            if str(p.get("id", "")) in (AGENT_SELF_ID, DEFAULT_PERSON_ID):
                continue
            if p.get("name") == name:
                return str(p["id"])
            raw = p.get("display_name", "") or ""
            aliases = [a.strip() for a in raw.replace(",", "、").split("、")]
            if name in aliases:
                return str(p["id"])
        return None

    def get_active_person_info(self) -> dict:
        """Kept for backward-compat — returns current speaker info."""
        return self.get_speaker_info() or {
            "id": AGENT_SELF_ID,
            "name": "__self__",
            "display_name": "Agent self",
        }
