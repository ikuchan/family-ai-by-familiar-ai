"""生成にかかった秒数を、呼び出しごとに必ずログへ残す（出-k-い）。

これまで所要時間を出していたのは調停だけで、主LLM・調べもの・整合チェック・声は
ログの時刻差から手で引くしかなかった。
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends.types import ToolCall, TurnResult
from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent

_LOG = "familiar_agent.loop.event_loop"


def _msgs(caplog):
    return [r.getMessage() for r in caplog.records]


def test_the_main_llm_logs_seconds_and_what_came_back(caplog) -> None:
    async def scenario():
        a = _agent(stream_returns=[])
        a.backend.stream_turn = AsyncMock(
            return_value=(
                TurnResult(
                    stop_reason="tool_use",
                    text="",
                    tool_calls=[ToolCall(id="t1", name="say", input={"text": "机が見えます"})],
                ),
                None,
            )
        )
        ip = InformationProcessing(a)
        with caplog.at_level(logging.INFO, logger=_LOG):
            await ip._run_main_llm(
                index=1,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "x"}, {"type": "image", "source": {}}],
                    }
                ],
                system="s",
                effort="high",
                capped=False,
                memories=[],
                w_id_map={},
                mem=None,
                recent_ctx="",
                retried=False,
            )
        await ip.close()

    asyncio.run(scenario())
    line = next(m for m in _msgs(caplog) if m.startswith("event-loop 主LLM "))
    assert "秒" in line and "effort=high" in line and "say" in line
    assert "6 字" in line and "写真=あり" in line


def test_a_lookup_logs_its_seconds(caplog) -> None:
    async def scenario():
        a = _agent(stream_returns=[])
        ip = InformationProcessing(a)
        with caplog.at_level(logging.INFO, logger=_LOG):
            await ip._run_lookup("recall", {"query": "昨日"}, "昨日", None, 1)
        await ip.close()

    asyncio.run(scenario())
    assert any(m.startswith("event-loop 調べもの recall") and "秒" in m for m in _msgs(caplog))


def test_the_coherence_check_logs_its_seconds(caplog) -> None:
    async def scenario():
        a = _agent(stream_returns=[])
        a.config.coherence_check = True
        a._evaluator.check_response_coherence = AsyncMock(return_value="x に反する")
        ip = InformationProcessing(a)
        with caplog.at_level(logging.INFO, logger=_LOG):
            await ip._coherence_violation("text", "", [])
        await ip.close()

    asyncio.run(scenario())
    assert any(
        m.startswith("event-loop 整合チェック") and "秒" in m and "違反=あり" in m
        for m in _msgs(caplog)
    )


def test_speaking_logs_its_seconds(caplog) -> None:
    from familiar_agent.io.dif import DIF

    tts = MagicMock()
    tts.call = AsyncMock(return_value="Said")
    d = DIF(tts=tts)
    with caplog.at_level(logging.INFO, logger="familiar_agent.io.dif"):
        asyncio.run(d.speak("机が見えます"))
    assert any(
        m.startswith("DIF 声") and "秒" in m and "6 字" in m
        for m in [r.getMessage() for r in caplog.records]
    )
