"""1 反復の分岐の特性テスト（環-ab E・2026-09-29）。

`_iterate` をメソッドオブジェクトへ移す前に、ループを通した試験が無かった分岐の結末を固める。
移したあとも同じ結末になることを確かめるためのもので、振る舞いを決め直すものではない。
"""

from __future__ import annotations


import asyncio
from unittest.mock import patch

from familiar_agent.loop import workspace
from familiar_agent.loop.event_loop import InformationProcessing
from tests._arbiter_fakes import jev_says, writer_says
from tests.test_event_loop import _agent


def _run_until(ip, done, *, push):
    """駆動体を起こし、`push()`（同期か非同期）で積み、`done()` まで待つ。"""

    async def scenario():
        ip.start()
        got = push()
        if asyncio.iscoroutine(got):
            await got
        for _ in range(300):
            if done():
                break
            await asyncio.sleep(0.005)
        await ip.close()

    asyncio.run(scenario())


def test_an_affect_request_the_arbiter_keeps_quiet_on_closes_silently(monkeypatch):
    """情動の求めで調停が「黙る」（light・言葉なし）と決めたら、主LLM を呼ばず沈黙で閉じる（出-w）。"""
    from familiar_agent.loop.arbiter import Arbiter, Decision

    a = _agent(stream_returns=[])
    a._jev = jev_says("light")
    a._utility_backend = writer_says({"text": ""})

    async def quiet(self, inp, on_decided=None):
        return Decision(branch="light", text="")

    # 段 5e（2026-10-10）から、表に無い軸は Jev に聞かずに主LLM へ行く（古い問いを外した）。ここで固めたいのは反復の
    # 側（調停が黙ると決めたら主LLM を呼ばず沈黙で閉じる）なので、調停の答えを黙るに固定する。
    monkeypatch.setattr(Arbiter, "decide", quiet)
    ip = InformationProcessing(a)
    outcomes: list[str] = []
    real_finish = ip._finish

    async def finish(text, memories, outcome, **kw):
        outcomes.append(outcome)
        return await real_finish(text, memories, outcome, **kw)

    ip._finish = finish  # type: ignore[method-assign]
    # 結果が届いた反復の黙るは test_arbiter_completion_by_rule が見る。
    _run_until(ip, lambda: outcomes, push=lambda: ip.push_affect("CALM", "落ち着いている"))
    assert outcomes == ["沈黙"]
    a.backend.stream_turn.assert_not_awaited()


def test_a_time_reference_recalls_again_from_that_time():
    """調停が時期を指したら（「去年の夏の話」）、その時点を基準に想起し直す。"""
    a = _agent(stream_returns=[])
    # 段 4-4c（2026-10-09）から、時期は recall を選んだときに語と一緒に軽量LLM が書く（別の問いにしない・本人）。
    a._jev = jev_says("action", action="recall")
    a._utility_backend = writer_says(
        {"query": "去年の夏", "time_ref": "2025-08-15T00:00:00", "time_span_days": 30}
    )
    ip = InformationProcessing(a)
    calls: list[dict] = []
    real_recall = workspace.recall

    async def recall(*args, **kw):
        calls.append(kw)
        return await real_recall(*args, **kw)

    with patch.object(workspace, "recall", recall):
        _run_until(
            ip, lambda: len(calls) >= 2, push=lambda: ip.push_utterance("去年の夏の話覚えてる？")
        )
    assert len(calls) == 2
    assert "time_ref" not in calls[0]
    assert calls[1]["time_span_days"] == 30
