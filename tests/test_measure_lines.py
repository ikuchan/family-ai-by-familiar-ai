"""層 3 が読む計測ログの行（記-a-に・2026-09-14）。

- `調停 秒=… 分岐=… 時間切れ=…`（`arbiter_timeout_sec` の材料）
- `直近 窓=n 端=<窓のいちばん古い起点> 外=<窓の外の次の起点>`（続き先の `相手` と突き合わせて窓 n を見直す材料）
- `関連 遠い=… 掘り=…`（関連想起の 2 並びのどちらから載ったか・`diffuse_far_share` の材料）
- `申告 important=… useless=… referred=… unused=…`（判定ごとの id 列）
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.core import measure


def _lines(tmp_path: Path, kind: str) -> list[str]:
    p = tmp_path / "rest_logs" / "measure.log"
    line_text = p.read_text(encoding="utf-8") if p.exists() else ""
    return [line for line in line_text.splitlines() if f" {kind} " in line]


def test_the_arbiter_records_seconds_branch_and_timeout(tmp_path):
    from familiar_agent.loop.arbiter import arbitrate

    measure.setup(base_dir=tmp_path)
    b = MagicMock(spec=["complete"])
    b.complete = AsyncMock(return_value='{"branch": "light", "text": "やあ"}')
    asyncio.run(arbitrate(b, utterance="こんにちは", workspace_ctx="", timeout=2.0))
    line = _lines(tmp_path, "調停")[-1]
    assert "分岐=light" in line and "時間切れ=no" in line and "秒=" in line

    async def slow(*_a, **_k):
        await asyncio.sleep(0.2)
        return "{}"

    b2 = MagicMock(spec=["complete"])
    b2.complete = AsyncMock(side_effect=slow)
    asyncio.run(arbitrate(b2, utterance="x", workspace_ctx="", timeout=0.01))
    assert "時間切れ=yes" in _lines(tmp_path, "調停")[-1]


def test_the_workspace_records_the_window_edge_and_the_next_origin_beyond_it(tmp_path):
    from datetime import datetime, timedelta, timezone

    from familiar_agent.io.oif import Said
    from familiar_agent.loop import workspace
    from familiar_agent.loop.request import Request

    measure.setup(base_dir=tmp_path)
    t0 = datetime(2026, 9, 14, 9, 0, tzinfo=timezone.utc)
    chains = {
        f"q{i}": [
            Said(
                obs_id=f"q{i}", content=str(i), role="起点", when=t0 + timedelta(minutes=i), depth=0
            )
        ]
        for i in range(5)
    }
    oif = MagicMock()
    oif.latest_origins = MagicMock(side_effect=lambda n: ["q4", "q3", "q2", "q1", "q0"][:n])
    oif.exchanges = MagicMock(side_effect=lambda o: chains[o])
    oif.actors = MagicMock(return_value={})
    oif.roles = MagicMock(return_value={})
    workspace.Workspace.build(oif, [], Request(), n_arbiter=2, n_main=3)
    line = _lines(tmp_path, "直近")[-1]
    assert "窓=3" in line and "端=q2" in line and "外=q1" in line


def test_verdicts_are_recorded_by_kind(tmp_path):
    from familiar_agent.loop import workspace

    measure.setup(base_dir=tmp_path)
    mem = MagicMock()
    workspace.apply_memory_verdicts(
        mem,
        [
            {"id": "aaaaaaaaaaaa", "verdict": "important"},
            {"id": "bbbbbbbbbbbb", "verdict": "unused"},
        ],
        {"aaaaaaaaaaaa": "aaaaaaaa-aaaa", "bbbbbbbbbbbb": "bbbbbbbb-bbbb"},
    )
    line = _lines(tmp_path, "申告")[-1]
    assert "important=aaaaaaaa-aaaa" in line and "unused=bbbbbbbb-bbbb" in line


def test_diffuse_recall_records_which_order_each_pick_came_from(tmp_path):
    from familiar_agent.core.diffuse import interleave_orders_tagged, note_orders

    measure.setup(base_dir=tmp_path)
    far, stale = ["a", "b", "c"], ["c", "d", "e"]
    note_orders(interleave_orders_tagged(far, stale, max_add=4, far_share=0.5))
    line = _lines(tmp_path, "関連")[-1]
    assert "遠い=a,b" in line and "掘り=c,d" in line
