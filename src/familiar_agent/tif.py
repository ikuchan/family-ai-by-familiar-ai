"""PI construction and PI->MI expansion (Phase 1 B-3, construction functions only).

`build_primitive` builds the boundary payload PI (`PrimitiveMentalItem`) from
mood M (`MoodPAD`, emotion) and drive D (`AiDrivers`, drive). `expand_to_mental`
performs I's intake-time PI->MI expansion, carrying the PI's emotion/drive
into a full `MentalItem` and adding the I-side attributes (id, content,
vector, supersedes, activation).

This is a thin vertical slice: these are pure construction functions, not
wired to actual firing, the I loop, recall, or desires. Nudge and N_PAD
(I->T mood modulation) are later work, dependent on recall/workspace (W)
and issue #11k.
"""

from __future__ import annotations

from .drive_register import AiDrivers
from .mood_register import MoodPAD
from .tools.memory import MentalItem, PrimitiveMentalItem


def build_primitive(emotion: MoodPAD, drive: AiDrivers) -> PrimitiveMentalItem:
    return PrimitiveMentalItem(emotion=emotion, drive=drive)


def expand_to_mental(
    pi: PrimitiveMentalItem, *, id: str, content: str,
    vector: object | None = None, supersedes: str | None = None,
    activation: float | None = None,
) -> MentalItem:
    return MentalItem(
        emotion=pi.emotion, drive=pi.drive,
        id=id, content=content, vector=vector,
        supersedes=supersedes, activation=activation,
    )
