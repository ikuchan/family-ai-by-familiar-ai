"""Default Mode Network processor — spontaneous memory recall during idle time.

Inspired by the brain's Default Mode Network (DMN), which activates when the
mind is not focused on external tasks.  The processor wanders through past
memories and surfaces associations as workspace Coalitions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tools.memory import ObservationMemory
    from .workspace import Coalition

logger = logging.getLogger(__name__)


class DefaultModeProcessor:
    """Generates spontaneous associations from stored memories.

    Parameters
    ----------
    memory:
        A memory backend exposing the ``recall_curiosities_async()`` coroutine.
    """

    def __init__(self, memory: ObservationMemory) -> None:
        self._memory = memory
        self._last_coalition: Coalition | None = None

    # ── Public API ────────────────────────────────────────────────────────

    async def wander(self) -> Coalition | None:
        """Recall memories and build a Coalition from the strongest one.

        Returns ``None`` when no memories are available.
        """
        from .workspace import Coalition

        memories = await self._memory.recall_curiosities_async()
        if not memories:
            self._last_coalition = None
            return None

        primary = memories[0]
        summary = primary.get("summary", "")
        importance = float(primary.get("confidence", 0.5))

        activation = max(0.0, min(1.0, importance))
        urgency = 0.1  # wandering is never urgent
        novelty = max(0.0, min(1.0, importance * 0.6))

        context_block = f"[DMN] Spontaneous recall: {summary}"

        coalition = Coalition(
            source="default_mode",
            summary=summary,
            activation=activation,
            urgency=urgency,
            novelty=novelty,
            context_block=context_block,
        )
        self._last_coalition = coalition
        return coalition

    def as_coalition(self) -> Coalition | None:
        """Return the most recent Coalition produced by :meth:`wander`, or ``None``."""
        return self._last_coalition
