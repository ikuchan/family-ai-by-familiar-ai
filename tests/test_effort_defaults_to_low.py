"""effort は既定 low・medium は限定列挙・high は人の明示だけ（出-k-ろ・課題5 G 章）。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.arbiter import _FALLBACK
from tests._arbiter_compat import _parse, arbitrate


def test_a_missing_effort_is_low() -> None:
    assert _parse('{"branch": "full"}').effort == "low"


def test_a_failed_arbitration_falls_to_low() -> None:
    assert _FALLBACK.effort == "low"
    b = MagicMock(spec=["complete"])
    b.complete = AsyncMock(return_value="???")
    assert asyncio.run(arbitrate(b, utterance="x", workspace_ctx="")).effort == "low"


def test_the_prompt_enumerates_medium_and_reserves_high() -> None:
    from familiar_agent.loop.arbiter import Arbiter, ArbiterInput

    # 深さは Jev の選択肢（出-au 段 5-7d）。low が「ほとんどの場合」、high は明示的に求められたときだけ。
    effort = Arbiter(jev=None, writer=None)._questions(
        ArbiterInput(utterance="x", workspace_ctx="")
    )["effort"]["criteria"]
    assert "ひと言で表せない複雑な気持ち" in effort["medium"]
    assert "4 つ以上の記憶" in effort["medium"]
    assert "調べた結果をまとめる" in effort["medium"]
    assert "明示的に" in effort["high"]
    assert "ほとんどの場合" in effort["low"]
