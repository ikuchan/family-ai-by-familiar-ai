"""取込と想起、版を書くのにかかった時間を残す（出-bz・2026-10-13・本人の決定）。

10/12 23:59 の天気の質問で、つなぎの反復が終わってから（36.72）完了の想起が W を組むまで（39.07）、2.35 秒ログに
何も出なかった。この間に入るのは、届いた結果の取込（版を O に書く・埋め込みと新しさを測る）と、完了の想起（手がかり
＝版の文の埋め込み）である。どこが重いかで、裏に出せるかどうかが変わるので、まず時間を残して分ける。
"""

from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock

from familiar_agent.loop import iteration
from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def test_a_timed_step_leaves_its_seconds(caplog):
    @iteration._timed("取込")
    async def step(self):
        await asyncio.sleep(0.05)
        return "済み"

    with caplog.at_level("INFO"):
        assert asyncio.run(step(None)) == "済み"
    line = next(r.getMessage() for r in caplog.records if "取込" in r.getMessage())
    assert re.search(r"event-loop 取込 0\.\d\d 秒", line)


def test_a_failed_step_still_leaves_its_seconds(caplog):
    """**反証の側**：途中で例外になっても、かかった時間は残す（例外はそのまま投げる）。"""

    @iteration._timed("想起")
    async def step(self):
        raise RuntimeError("落ちた")

    with caplog.at_level("INFO"):
        try:
            asyncio.run(step(None))
        except RuntimeError:
            pass
        else:
            raise AssertionError("例外が消えた")
    assert "event-loop 想起" in caplog.text


def test_the_iteration_times_intake_and_recall():
    assert iteration.Iteration._intake._timed_label == "取込"
    assert iteration.Iteration._recall._timed_label == "想起"


def test_writing_a_version_leaves_its_seconds_and_size(caplog):
    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._agent._oif.write = AsyncMock(return_value="v-1")
    ip._req.request_id = "req-1"
    ip._req.request_text = "明日の天気は？"

    async def go():
        await ip._write_version()
        await ip.close()

    with caplog.at_level("INFO"):
        asyncio.run(go())
    line = next(r.getMessage() for r in caplog.records if "版を書いた" in r.getMessage())
    assert re.search(r"版を書いた \d+\.\d\d 秒（\d+ 字）", line)
