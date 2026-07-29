"""ワークスペース候補（Coalition）の採点。

競合と放送を行っていた GlobalWorkspace は #12a で撤去した。
"""

from __future__ import annotations


import pytest

from familiar_agent.workspace import Coalition


# ── Coalition ─────────────────────────────────────────────────────────────────


def test_coalition_score_basic():
    c = Coalition(
        source="desire",
        summary="want to look around",
        activation=1.0,
        urgency=1.0,
        novelty=1.0,
        context_block="(desire look_around)",
    )
    assert c.score() == pytest.approx(1.0)


def test_coalition_score_zero_activation():
    c = Coalition(
        source="memory",
        summary="nothing",
        activation=0.0,
        urgency=1.0,
        novelty=1.0,
        context_block="",
    )
    assert c.score() == pytest.approx(0.0)


def test_coalition_score_weights():
    # score = activation * (0.4*urgency + 0.3*novelty + 0.3*1.0)
    c = Coalition(
        source="scene",
        summary="person appeared",
        activation=0.8,
        urgency=1.0,
        novelty=0.0,
        context_block="...",
    )
    expected = 0.8 * (0.4 * 1.0 + 0.3 * 0.0 + 0.3)
    assert c.score() == pytest.approx(expected)
