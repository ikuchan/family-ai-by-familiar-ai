"""調停の時間切れ（記-a の前に片付けた申し送り）。

「黙って」と頼まれたときの調停は実測 4.18 秒かかり（普通の会話は 0.93〜1.10 秒）、
時間切れ 2.0 秒では届かずフルへ倒れていた。沈黙依頼が読まれないまま素通りする。
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from familiar_agent.config import AgentConfig
from familiar_agent.loop.arbiter import arbitrate


def test_the_timeout_is_five_seconds_by_default():
    """実測 4.18 秒に 0.8 秒の余裕を見た値。"""
    with patch.dict(os.environ, {}, clear=True):
        assert AgentConfig().arbiter_timeout_sec == pytest.approx(5.0)


def test_the_timeout_can_be_set_by_env():
    with patch.dict(os.environ, {"ARBITER_TIMEOUT_SEC": "3.5"}, clear=True):
        assert AgentConfig().arbiter_timeout_sec == pytest.approx(3.5)


def test_a_reply_that_arrives_within_the_timeout_is_used():
    """4.2 秒で返る調停（「黙って」の実測に近い）を、時間切れにしない。"""
    backend = MagicMock()

    async def _slow(prompt, max_tokens, *, system=None):
        await asyncio.sleep(0.05)
        return '{"branch":"light","text":"わかった"}'

    backend.complete = AsyncMock(side_effect=_slow)
    decision = asyncio.run(arbitrate(backend, utterance="黙って", workspace_ctx="",
                                     timeout=0.5))
    assert decision.branch == "light"


def test_a_reply_that_exceeds_the_timeout_falls_back_to_full():
    """届かなければフルへ倒す（従来どおり）。"""
    backend = MagicMock()

    async def _too_slow(prompt, max_tokens):
        await asyncio.sleep(0.3)
        return '{"branch":"light","text":"間に合わない"}'

    backend.complete = AsyncMock(side_effect=_too_slow)
    decision = asyncio.run(arbitrate(backend, utterance="黙って", workspace_ctx="",
                                     timeout=0.05))
    assert decision.branch == "full"
