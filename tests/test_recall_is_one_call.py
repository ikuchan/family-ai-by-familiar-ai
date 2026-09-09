"""想起して W を組む手順を、器へ1つにする（環-e-に・に-5-ろ）。

`_iterate` は同じ呼び出しを**2度**書いていた。

1. 反復の頭（いまが基準）
2. 調停が時期を指したときの引き直し（「去年の夏の話」）

どちらも `recall_async(cue, n, min_score, weights, open_ids)` ＋ `_compose_workspace` で、
違うのは時期の2欄だけである。片方を直してもう片方を忘れれば、**基準を移した反復だけ床が
効かない**といった食い違いが黙って入る（床＝`min_score` は実際に、連想想起には渡っていて
イベントループにだけ渡っていなかった）。

**重みは呼び手が持つ。** `jitter_weights` は乱数を足すので、器の中で作り直すと引き直しの
たびに別の重みになる。同じ反復のあいだは同じ重みでなければならない。

**挙動は変えない。**
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.event_loop import InformationProcessing


def _ip():
    ip = InformationProcessing.__new__(InformationProcessing)
    ip._open_ids = MagicMock(return_value=["o1"])
    ip._compose_workspace = MagicMock(return_value="[作業状態]")
    mem = MagicMock()
    mem.recall_async = AsyncMock(return_value=[{"memory_id": "m1"}])
    return ip, mem


# ── 器 ─────────────────────────────────────────────────────────────────────


def test_it_returns_both_the_records_and_the_workspace():
    """W は記録から組むので、**2つで1つ**である。別々に呼べば片方だけ古くなる。"""
    ip, mem = _ip()
    memories, workspace = asyncio.run(ip._recall_workspace(mem, "手がかり", weights=None))
    assert memories == [{"memory_id": "m1"}]
    assert workspace == "[作業状態]"
    ip._compose_workspace.assert_called_once_with(mem, [{"memory_id": "m1"}])


def test_the_caller_owns_the_weights():
    """器は重みを作らない（作れば引き直しのたびに jitter が振り直される）。"""
    params = inspect.signature(InformationProcessing._recall_workspace).parameters
    assert "weights" in params
    # **呼び出しを見る**（語ではない）。docstring がなぜ作らないかを述べているので、
    # 語で探すと自分の説明文に当たる。
    src = inspect.getsource(InformationProcessing._recall_workspace)
    assert "jitter_weights(" not in src
    assert "recall_weights(" not in src


def test_the_floor_is_always_passed():
    """床（min_score）は両方の呼び出しに効く。2度書きだと片方が抜ける。"""
    ip, mem = _ip()
    asyncio.run(ip._recall_workspace(mem, "手がかり", weights=None))
    assert mem.recall_async.await_args.kwargs["min_score"] is not None


def test_a_time_reference_can_be_moved():
    ip, mem = _ip()
    asyncio.run(
        ip._recall_workspace(mem, "手がかり", weights=None, time_ref=1.0, time_span_days=30.0)
    )
    kw = mem.recall_async.await_args.kwargs
    assert kw["time_ref"] == 1.0
    assert kw["time_span_days"] == 30.0


def test_without_a_time_reference_the_present_is_the_basis():
    ip, mem = _ip()
    asyncio.run(ip._recall_workspace(mem, "手がかり", weights=None))
    kw = mem.recall_async.await_args.kwargs
    assert kw["time_ref"] is None
    assert kw["time_span_days"] is None


# ── 呼び手 ─────────────────────────────────────────────────────────────────


def test_the_iteration_does_not_recall_by_hand_anymore():
    src = inspect.getsource(InformationProcessing._iterate)
    assert "recall_async(" not in src, "反復が自分で想起している"
    assert "_compose_workspace(" not in src, "反復が自分で W を組んでいる"


def test_the_iteration_got_shorter():
    src = inspect.getsource(InformationProcessing._iterate)
    assert len(src.split("\n")) <= 225, "薄くなっていない"
