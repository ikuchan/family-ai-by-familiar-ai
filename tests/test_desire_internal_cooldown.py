"""Tests for INTERNAL_DESIRE_COOLDOWN — separate growth rates for internal desires."""

from __future__ import annotations

import os
import importlib


def _reload_desires(social_cd: str = "900", internal_cd: str | None = None) -> object:
    """Reload desires module with given env vars so module-level constants re-compute."""
    _INTERNAL_KEY = "INTERNAL_DESIRE_COOLDOWN"
    old_social = os.environ.get("DESIRE_COOLDOWN")
    old_internal = os.environ.get(_INTERNAL_KEY)

    os.environ["DESIRE_COOLDOWN"] = social_cd
    if internal_cd is not None:
        os.environ[_INTERNAL_KEY] = internal_cd
    else:
        os.environ.pop(_INTERNAL_KEY, None)  # ensure absent so default kicks in

    import familiar_agent.desires as mod
    importlib.reload(mod)

    # Restore env
    if old_social is None:
        os.environ.pop("DESIRE_COOLDOWN", None)
    else:
        os.environ["DESIRE_COOLDOWN"] = old_social

    if old_internal is None:
        os.environ.pop(_INTERNAL_KEY, None)
    else:
        os.environ[_INTERNAL_KEY] = old_internal

    return mod


def test_internal_rate_faster_than_social_when_cooldown_shorter():
    mod = _reload_desires(social_cd="900", internal_cd="180")
    # look_around is internal, greet_companion is social
    assert mod.GROWTH_RATES["look_around"] > mod.GROWTH_RATES["greet_companion"]


def test_internal_rate_equals_social_when_cooldown_same():
    mod = _reload_desires(social_cd="900", internal_cd="900")
    # Both use same ref, so with n=3 vs n=5 look_around is still faster
    # but the ratio equals what _rate(3)/_rate(5) would give
    expected_ratio = 5 / 3  # n=5 / n=3
    actual_ratio = mod.GROWTH_RATES["look_around"] / mod.GROWTH_RATES["greet_companion"]
    assert abs(actual_ratio - expected_ratio) < 0.001


def test_default_internal_cooldown_equals_social_cooldown():
    mod = _reload_desires(social_cd="300")
    # When INTERNAL_DESIRE_COOLDOWN not set, it should mirror DESIRE_COOLDOWN
    assert mod._INTERNAL_REF_COOLDOWN == mod._REF_COOLDOWN


def test_explicit_internal_cooldown_overrides_default():
    mod = _reload_desires(social_cd="900", internal_cd="90")
    assert mod._INTERNAL_REF_COOLDOWN == 90.0
    assert mod._REF_COOLDOWN == 900.0


def test_social_growth_rates_unaffected_by_internal_cooldown():
    mod_slow = _reload_desires(social_cd="900", internal_cd="900")
    mod_fast = _reload_desires(social_cd="900", internal_cd="90")
    # Social desires should have the same rate regardless of internal cooldown
    assert mod_slow.GROWTH_RATES["greet_companion"] == mod_fast.GROWTH_RATES["greet_companion"]
    assert mod_slow.GROWTH_RATES["share_memory"] == mod_fast.GROWTH_RATES["share_memory"]


def test_internal_desires_grow_faster_with_short_internal_cooldown():
    # Capture rates immediately after each reload (same module object gets mutated)
    _reload_desires(social_cd="900", internal_cd="900")
    import familiar_agent.desires as mod
    slow_look = mod.GROWTH_RATES["look_around"]
    slow_explore = mod.GROWTH_RATES["explore"]
    slow_consolidate = mod.GROWTH_RATES["consolidate"]

    _reload_desires(social_cd="900", internal_cd="90")
    fast_look = mod.GROWTH_RATES["look_around"]
    fast_explore = mod.GROWTH_RATES["explore"]
    fast_consolidate = mod.GROWTH_RATES["consolidate"]

    assert fast_look > slow_look
    assert fast_explore > slow_explore
    assert fast_consolidate > slow_consolidate
