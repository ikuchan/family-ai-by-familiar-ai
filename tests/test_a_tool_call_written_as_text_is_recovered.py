"""主LLM が道具呼び出しを文で書いたら、呼び出しとして拾う（2026-09-13 実機で露見）。

「今日の予定は？」で Sonnet 5 が `<invoke name="recall"><parameter name="query">…</parameter></invoke>`
を**本文として**返し（tool_use ブロック無し）、ループは素テキストとして扱って画面と O に
そのまま出した。プロンプトにこの形は無く、モデル側の取りこぼしだが、そのまま人に見せる
ものではない。
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

from familiar_agent.backends.types import TurnResult
from familiar_agent.core.tool_text import tool_calls_from_text
from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent

_XML = (
    '<invoke name="recall">\n<parameter name="query">2026-09-13 家族の予定</parameter>\n'
    '<parameter name="n">5</parameter>\n</invoke>'
)


def test_the_xml_becomes_a_tool_call() -> None:
    calls = tool_calls_from_text(_XML)
    assert len(calls) == 1
    assert calls[0].name == "recall"
    assert calls[0].input == {"query": "2026-09-13 家族の予定", "n": 5}


def test_plain_text_is_left_alone() -> None:
    assert tool_calls_from_text("今日は晴れだよ") == []


def test_the_loop_recovers_and_logs(caplog) -> None:
    async def scenario():
        a = _agent(stream_returns=[])
        a.backend.stream_turn = AsyncMock(
            return_value=(TurnResult(stop_reason="end_turn", text=_XML), None)
        )
        ip = InformationProcessing(a)
        with caplog.at_level(logging.INFO, logger="familiar_agent.loop.event_loop"):
            await ip._run_main_llm(
                index=1,
                messages=[],
                system="s",
                effort="low",
                capped=False,
                memories=[],
                w_id_map={},
                mem=None,
                recent_ctx="",
                retried=False,
                max_tokens=500,
            )
        item = ip._triggers.get_nowait()
        await ip.close()
        return item.decision.result

    result = asyncio.run(scenario())
    assert [tc.name for tc in result.tool_calls] == ["recall"]
    assert result.text == ""
    assert any("文で書いた" in r.getMessage() for r in caplog.records)
