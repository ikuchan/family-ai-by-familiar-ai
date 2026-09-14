"""REST 内省・層 1 の「$I$ を測る → $\\Delta$ 減らす」（`loop/rest_info.py`・記-a-ろ-ろ）。

店はモック。設定値（$HL$・$w_t$・$w_g$・$I^\\*$）は `agent.config` から読む。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

from familiar_agent.core import info_measure as im
from familiar_agent.core import measure
from familiar_agent.loop import rest_info

NOW = datetime(2026, 9, 15, 3, 0, tzinfo=timezone.utc)


def _agent(core_rows, fresh_rows, *, target=2.0**17, decayed=0):
    a = MagicMock()
    a.config.memory.recall_half_life_days = 10.0
    a.config.memory.recall_w_t = 1.0
    a.config.memory.recall_w_g = 1.5
    a.config.memory.info_target_bits = target
    a._oif.core_faces = MagicMock(return_value=core_rows)
    a._oif.fresh_since_last_rest = MagicMock(return_value=fresh_rows)
    a._oif.decay_groundedness = MagicMock(return_value=decayed)
    return a


def _row(chars: int, days: float, n: int, person: str = "self"):
    return {
        "obs_id": f"o{chars}",
        "person_id": person,
        "chars": chars,
        "timestamp": NOW - timedelta(days=days),
        "groundedness_g0": 1.0,
        "groundedness_n": n,
    }


def _lines(tmp_path: Path) -> list[str]:
    return [
        f" {r.kind} " + " ".join(f"{k}={v}" for k, v in r.fields.items())
        for r in measure.read_rows(base_dir=tmp_path)
    ]


def test_below_target_measures_and_does_not_decay(tmp_path) -> None:
    measure.setup(base_dir=tmp_path)
    core = [_row(100, 0.0, 10**6), _row(50, 10.0, 1, "p1")]
    fresh = [_row(30, 0.1, 0)]
    a = _agent(core, fresh)
    text = asyncio.run(rest_info.measure_and_decay(a, now=NOW))
    a._oif.decay_groundedness.assert_not_called()
    expected = im.total(core, now=NOW, hl=10.0, w_t=1.0, w_g=1.5)
    assert f"{int(expected)}" in text and "減らさなかった" in text
    line = next(ln for ln in _lines(tmp_path) if " 層1計測 " in ln)
    assert f"I={int(expected)}" in line and "核=2" in line and "今日=1" in line and "面=2" in line
    assert not any(" 層1減り " in ln for ln in _lines(tmp_path))


def test_above_target_decays_by_delta_and_records_it(tmp_path) -> None:
    measure.setup(base_dir=tmp_path)
    core = [_row(100_000, 0.0, 10**6)]  # 550 kbit ≫ 2^17 → Δ=ceil(log2(4.2))=3
    a = _agent(core, [], decayed=7)
    text = asyncio.run(rest_info.measure_and_decay(a, now=NOW))
    a._oif.decay_groundedness.assert_called_once_with(3)
    assert "Δ=3" in text and "7 件" in text
    line = next(ln for ln in _lines(tmp_path) if " 層1減り " in ln)
    assert "Δ=3" in line and "動かした=7" in line


def test_the_target_comes_from_the_layer_three_setting() -> None:
    from familiar_agent.config import MemoryConfig
    from familiar_agent.core import settings

    assert MemoryConfig().info_target_bits == 2.0**17
    s = settings.get("MemoryConfig.info_target_bits")
    assert s is not None and s.lo <= 2.0**17 <= s.hi and s.measure_kind == "層1"
