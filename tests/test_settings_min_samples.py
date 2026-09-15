"""層 3 は標本が足りなければ動かさない（記-k・2026-09-15）。

改名された計測ログ 121 行のうち `気分`／`欲求` は 4 行だったのに、内部状態の境目 9 件が動いた。
規則ごとに最小の標本数〔仮〕：内部状態 50 行・窓 20 行・時間切れ 20 行・関連 20 行。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.core import measure
from familiar_agent.loop import rest_settings as rs

NOW = datetime(2026, 9, 15, 3, 0, tzinfo=timezone.utc)


def _rows(kind: str, n: int, **fields) -> list[measure.Row]:
    return [
        measure.Row(when=NOW, kind=kind, fields={k: str(v) for k, v in fields.items()})
        for _ in range(n)
    ]


def test_summaries_carry_their_sample_counts():
    inner = measure.summarize_inner_state(
        _rows("気分", 4, P=0.3, Pn=0.1, A=0.4, Dom=0.5) + _rows("欲求", 3, seeking=0.6)
    )
    assert inner["件数"] == {"気分": 4, "欲求": 3}
    win = measure.summarize_window(
        _rows("直近", 1, 端="a", 外="b") + _rows("続き先", 5, 結末="続き", 相手="a")
    )
    assert win["続き"] == 5
    rel = measure.summarize_relation(
        _rows("関連", 1, 遠い="x", 掘り="y") + _rows("申告", 1, important="x", referred="-")
    )
    assert rel["件数"] == 1


def test_inner_state_does_not_move_below_the_minimum():
    few = measure.summarize_inner_state(
        _rows("気分", rs.MIN_SAMPLES["内部状態"] - 1, P=0.9, Pn=0.0, A=0.9, Dom=0.9)
    )
    assert rs.adjust_inner_state(few) == []
    enough = measure.summarize_inner_state(
        _rows("気分", rs.MIN_SAMPLES["内部状態"], P=0.9, Pn=0.0, A=0.9, Dom=0.9)
    )
    assert rs.adjust_inner_state(enough) != []


def test_window_does_not_move_below_the_minimum():
    few = measure.summarize_window(
        _rows("直近", 1, 端="a", 外="b")
        + _rows("続き先", rs.MIN_SAMPLES["窓"] - 1, 結末="続き", 相手="b")
    )
    assert rs.adjust_window(few) == []
    enough = measure.summarize_window(
        _rows("直近", 1, 端="a", 外="b")
        + _rows("続き先", rs.MIN_SAMPLES["窓"], 結末="続き", 相手="b")
    )
    assert rs.adjust_window(enough) != []


def test_far_share_does_not_move_below_the_minimum():
    few = {"遠い": 3, "掘り": 0, "件数": rs.MIN_SAMPLES["関連"] - 1}
    assert rs.adjust_far_share(few) == []
    enough = {"遠い": 3, "掘り": 0, "件数": rs.MIN_SAMPLES["関連"]}
    assert rs.adjust_far_share(enough) != []


def test_timeout_is_not_asked_below_the_minimum():
    agent = MagicMock()
    agent.backend.complete = AsyncMock(return_value='{"arbiter_timeout_sec": 6.0, "reason": "x"}')
    few = {
        "件数": rs.MIN_SAMPLES["時間切れ"] - 1,
        "中央": 1.0,
        "p90": 2.0,
        "最大": 3.0,
        "時間切れ": 0,
        "時間切れの割合": 0.0,
    }
    assert asyncio.run(rs.adjust_timeout(agent, few)) == []
    agent.backend.complete.assert_not_called()


def test_the_minimums_are_the_agreed_values():
    assert rs.MIN_SAMPLES == {"内部状態": 50, "窓": 20, "時間切れ": 20, "関連": 20}
