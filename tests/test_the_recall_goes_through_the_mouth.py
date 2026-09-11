"""**想起も記憶の口を通る**（環-e-い・呼び替え）。

口が正しい面を引けるようにしたのが前段（`View.viewpoint`）。ここでは呼び手を移す。

**器が変わる。** いまの想起は辞書の並びを返し、5つの読み手がその形に乗っていた——
`compose`（W を組む）・W の1行・`facts_ctx`（整合チェックの材料）・共起・想起のログ。
`Recalled` は `MI` ＋そのときの採点で、**store の語（`memory_id`・`summary`）を外へ出さない**。

**`confidence` は `Recalled` へ足す。** `fit`・`groundedness` と同じ「そのときの採点」で、
保存する値ではない（`MI` は保存する値だけを持つ）。

**W の1行を組むのは核（`workspace`）の仕事にする。** `ObservationMemory.format_for_context`
に置いたままだと、記憶が OIF の器を知ることになる（依存が逆向き）。`agent` 側の呼び手は
辞書のままなので、そちらの面は残す。
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.io.oif import MI, Recalled
from familiar_agent.loop import workspace
from familiar_agent.loop.request import Request


def _r(obs_id="m1", content="運動会の話", fit=0.7, conf=0.8, direction="発話"):
    return Recalled(
        mi=MI(
            id=f"facet-{obs_id}",
            obs_id=obs_id,
            content=content,
            timestamp=datetime(2026, 9, 11, 15, 0),
            direction=direction,
        ),
        fit=fit,
        groundedness=1.0,
        confidence=conf,
    )


def test_the_score_carries_the_confidence():
    """`confidence` は採点の一部。`MI` ではなく `Recalled` が持つ。"""
    assert "confidence" in Recalled.__dataclass_fields__
    assert "confidence" not in MI.__dataclass_fields__


def test_the_workspace_builds_its_own_line():
    """W の1行は核が組む。誰の記録かも言う（記-f）。"""
    oif = MagicMock()
    oif.actors.return_value = {"m1": "ゆうすけ"}
    text, id_map = workspace.compose(oif, [_r()], Request())
    assert "ゆうすけが言った: 運動会の話" in text
    assert "id:m1" in text and "適合度:0.70" in text and "conf:0.80" in text
    assert id_map == {"m1": "m1"}
    oif.format_for_context.assert_not_called()  # 記憶の面は使わない


def test_a_low_confidence_record_is_marked():
    text, _ = workspace.compose(
        MagicMock(actors=MagicMock(return_value={})), [_r(conf=0.4)], Request()
    )
    assert "low-confidence" in text


def test_the_recall_asks_the_mouth_with_the_speakers_face():
    """**話者の面から引く。** 口が持つのは基底の記憶なので、視点を渡さなければ
    `__self__` の面に移ってしまう（記-f の直しを逆向きに壊す）。"""
    oif = MagicMock(recall=AsyncMock(return_value=[_r()]))
    got, text, id_map = asyncio.run(
        workspace.recall(oif, "手がかり", viewpoint="ゆうすけ", weights=None, req=Request())
    )
    cue, view = oif.recall.await_args.args
    assert cue.text == "手がかり"
    assert view.viewpoint == "ゆうすけ"
    assert [r.mi.obs_id for r in got] == ["m1"]


def test_the_loop_no_longer_recalls_from_the_memory_directly():
    import pathlib

    w = (
        pathlib.Path(__file__).resolve().parent.parent
        / "src"
        / "familiar_agent"
        / "loop"
        / "workspace.py"
    ).read_text(encoding="utf-8")
    assert "mem.recall_async(" not in w, "記憶から直に想起している"
    assert "_oif.recall(" in w or "oif.recall(" in w


def test_the_agents_own_formatter_stays():
    """`agent` 側の呼び手は辞書のままなので、記憶の面は残す。"""
    from familiar_agent.tools.memory import ObservationMemory

    assert callable(ObservationMemory.format_for_context)
    src = inspect.getsource(ObservationMemory.format_for_context)
    assert "m['summary']" in src or 'm["summary"]' in src
