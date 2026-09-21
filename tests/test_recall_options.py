"""思い出し方の指定（出-ah・2026-09-21）。純関数。

W の「過去の記憶」に何が載るかを決める軸のうち、**視点・件数と思い出し方・時期と幅・直近の窓**を
主LLM が動かせるようにする。ここはその丸めと割り当てだけを持つ（引くのは `workspace`）。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from familiar_agent.config import RecallWeights
from familiar_agent.core.recall_options import (
    MAX_K,
    RECENT_MAX_MIN,
    RECENT_MAX_TURNS,
    WAYS,
    clamp_k,
    clamp_recent,
    parse_when,
    weights_for,
)

BASE = RecallWeights(w_r=1.0, w_t=1.0, w_e=1.0, w_g=1.5, w_p=1.0)


def test_the_named_way_doubles_its_axis_and_thins_the_others():
    w = weights_for("新しい順に", BASE)
    assert w.w_t == pytest.approx(2.0)  # 時間を 2 倍
    assert w.w_r == pytest.approx(1.0 * 2 / 3)
    assert w.w_g == pytest.approx(1.5 * 2 / 3)


def test_every_way_points_at_one_axis():
    assert set(WAYS) == {
        "新しい順に",
        "印象に残っていることを",
        "よく思い出すことを",
        "この人との関わりで",
        "話に近いものを",
    }
    assert weights_for("印象に残っていることを", BASE).w_e == pytest.approx(2.0)
    assert weights_for("よく思い出すことを", BASE).w_g == pytest.approx(3.0)
    assert weights_for("この人との関わりで", BASE).w_p == pytest.approx(2.0)
    assert weights_for("話に近いものを", BASE).w_r == pytest.approx(2.0)


def test_an_unknown_way_changes_nothing():
    assert weights_for("てきとうに", BASE) == BASE
    assert weights_for("", BASE) == BASE


def test_the_count_is_clamped():
    assert MAX_K == 20
    assert clamp_k(30) == 20
    assert clamp_k(0) == 1
    assert clamp_k(None) == 20  # 指定なしは上限まで（深く思い出す道具なので）
    assert clamp_k(7) == 7


def test_the_recent_window_is_clamped():
    assert (RECENT_MAX_MIN, RECENT_MAX_TURNS) == (30, 20)
    assert clamp_recent(90, 50) == (30, 20)
    assert clamp_recent(10, 5) == (10, 5)
    assert clamp_recent(0, 0) == (1, 1)
    assert clamp_recent(None, None) == (30, 20)


def test_a_date_is_read_with_its_span():
    got = parse_when("2026-08-15", 45)
    assert got is not None
    when, span = got
    assert when == datetime(2026, 8, 15, tzinfo=timezone.utc).timestamp()
    assert span == 45.0


def test_a_date_that_cannot_be_read_is_refused():
    assert parse_when("去年の夏", 30) is None
    assert parse_when("", 30) is None


def test_the_span_has_a_default_and_is_not_negative():
    got = parse_when("2026-08-15", None)
    assert got is not None and got[1] == 30.0
    got = parse_when("2026-08-15", -5)
    assert got is not None and got[1] == 1.0
