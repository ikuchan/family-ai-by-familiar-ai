"""ワークスペースへの候補（Coalition）の器。

各処理系が「W に載せたいもの」を同じ形で差し出すための dataclass を置く。競合と
放送を行っていた `GlobalWorkspace` は #12a で撤去した（登録した listener は発火せず、
競合の結果を読む経路も無くなっていた）。いまの W は `loop/event_loop.py` の
`_compose_workspace` が組み立てる。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Minimum ignition threshold floor (prevents runaway suppression)
_MIN_THRESHOLD = 0.05

# How much each unit of prediction error lowers the threshold
_ERROR_SENSITIVITY = 0.15


@dataclass
class Coalition:
    """A candidate for workspace access from one specialized processor.

    Attributes:
        source: Name of the originating processor (e.g. "desire", "scene").
        summary: Short natural-language description of the content.
        dynamism: Base strength from the source processor (0.0–1.0).
        urgency: Time-sensitivity (e.g. "person appeared" = high urgency).
        novelty: How unexpected this content is (connects to prediction error).
        context_block: Formatted text for LLM prompt injection if this wins.
    """

    source: str
    summary: str
    dynamism: float
    urgency: float
    novelty: float
    context_block: str

    def score(self) -> float:
        """Composite score used in workspace competition.

        score = dynamism × (0.4×urgency + 0.3×novelty + 0.3)

        The constant 0.3 ensures coalitions with zero urgency/novelty can
        still compete if dynamism is high enough.
        """
        return self.dynamism * (0.4 * self.urgency + 0.3 * self.novelty + 0.3)
