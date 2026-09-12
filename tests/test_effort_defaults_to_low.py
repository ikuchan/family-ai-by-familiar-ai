"""effort は既定 low・medium は限定列挙・high は人の明示だけ（出-k-ろ・課題5 G 章）。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.arbiter import _FALLBACK, _parse, arbitrate


def test_a_missing_effort_is_low() -> None:
    assert _parse('{"branch": "full"}').effort == "low"


def test_a_failed_arbitration_falls_to_low() -> None:
    assert _FALLBACK.effort == "low"
    b = MagicMock(spec=["complete"])
    b.complete = AsyncMock(return_value="???")
    assert asyncio.run(arbitrate(b, utterance="x", workspace_ctx="")).effort == "low"


def test_the_prompt_enumerates_medium_and_reserves_high() -> None:
    b = MagicMock(spec=["complete"])
    b.complete = AsyncMock(return_value='{"branch": "full"}')
    asyncio.run(arbitrate(b, utterance="x", workspace_ctx=""))
    prompt = b.complete.call_args.args[0]
    assert "ひと言で表せない複雑な気持ち" in prompt
    assert "4 つ以上の記憶" in prompt
    assert "調べた結果をまとめる" in prompt
    assert "明示的に" in prompt and '"high"' in prompt
    assert '既定は "low"' in prompt
