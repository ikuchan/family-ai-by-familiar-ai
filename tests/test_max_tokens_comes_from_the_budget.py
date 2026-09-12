"""主LLM の `max_tokens` は予算から来る（出-k-ろ）。切れたらログに残す。"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

from familiar_agent.backends.types import TurnResult
from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def _run(max_tokens: int, stop_reason: str, caplog):
    async def scenario():
        a = _agent(stream_returns=[])
        a.backend.stream_turn = AsyncMock(
            return_value=(TurnResult(stop_reason=stop_reason, text="x"), None)
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
                max_tokens=max_tokens,
            )
        await ip.close()
        return a.backend.stream_turn.call_args.kwargs["max_tokens"]

    return asyncio.run(scenario())


def test_the_given_max_tokens_reaches_the_backend(caplog) -> None:
    assert _run(700, "end_turn", caplog) == 700


def test_a_cut_reply_is_logged(caplog) -> None:
    _run(120, "max_tokens", caplog)
    assert any(
        "max_tokens" in r.getMessage() and "切れた" in r.getMessage() for r in caplog.records
    )


def test_the_iteration_dispatches_with_the_budget() -> None:
    import inspect

    src = inspect.getsource(InformationProcessing._iterate)
    assert "reply_budget.decide(" in src or "decide(" in src
    assert "max_tokens=budget.max_tokens" in src
