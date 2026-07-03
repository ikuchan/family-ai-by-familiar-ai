"""Tests for PI construction and PI->MI expansion (Phase 1 B-3, construction functions only).

tif is a new, unconnected module: build_primitive/expand_to_mental are pure
construction functions, not wired to firing/loop/recall/desires in this step.
Nudge and N_PAD are later work.
"""

from __future__ import annotations

from familiar_agent.mood_register import MoodPAD
from familiar_agent.drive_register import AiDrivers
from familiar_agent.tools.memory import PrimitiveMentalItem, MentalItem
from familiar_agent.tif import build_primitive, expand_to_mental


# ── build_primitive carries M and D as-is (identity, not a copy) ────────────

def test_build_primitive_carries_m_and_d() -> None:
    m = MoodPAD(p=0.7, pn=0.2, a=0.6, dom=0.4)
    d = AiDrivers(seeking=0.3, bond=0.5)
    pi = build_primitive(m, d)
    assert isinstance(pi, PrimitiveMentalItem)
    assert pi.emotion is m and pi.drive is d


# ── expand_to_mental inherits PI's emotion/drive and adds I-side attributes ──

def test_expand_to_mental_inherits_and_adds() -> None:
    m = MoodPAD(p=0.7)
    d = AiDrivers(seeking=0.3)
    pi = build_primitive(m, d)
    mi = expand_to_mental(pi, id="obs-1", content="こんにちは", activation=0.75)
    assert isinstance(mi, MentalItem)
    assert mi.emotion is m and mi.drive is d
    assert mi.id == "obs-1" and mi.content == "こんにちは" and mi.activation == 0.75
    assert mi.vector is None and mi.supersedes is None


# ── MentalItem is a PrimitiveMentalItem subclass (emotion/drive present) ────

def test_mental_is_primitive_subclass() -> None:
    mi = expand_to_mental(build_primitive(MoodPAD(), AiDrivers()), id="x", content="y")
    assert isinstance(mi, PrimitiveMentalItem)
    assert hasattr(mi, "emotion") and hasattr(mi, "drive")
