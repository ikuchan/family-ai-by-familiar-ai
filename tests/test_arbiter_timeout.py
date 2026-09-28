"""調停の時間切れ（記-a の前に片付けた申し送り）。

「黙って」と頼まれたときの調停は実測 4.18 秒かかり（普通の会話は 0.93〜1.10 秒）、
時間切れ 2.0 秒では届かずフルへ倒れていた。沈黙依頼が読まれないまま素通りする。
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import pytest

from familiar_agent.config import AgentConfig
from tests._arbiter_fakes import decide, jev_says, writer_says


def test_the_timeout_is_five_seconds_by_default():
    """実測 4.18 秒に 0.8 秒の余裕を見た値。"""
    with patch.dict(os.environ, {}, clear=True):
        assert AgentConfig().arbiter_timeout_sec == pytest.approx(5.0)


def test_the_timeout_is_a_layer_three_setting_and_ignores_env():
    """層 3 の設定値（DB > 既定・記-a-に・2026-09-14）。`ARBITER_TIMEOUT_SEC` は読まない。"""
    from familiar_agent import config_overrides as co

    co._delete_all()
    co.clear_cache()
    with patch.dict(os.environ, {"ARBITER_TIMEOUT_SEC": "3.5"}, clear=True):
        assert AgentConfig().arbiter_timeout_sec == pytest.approx(5.0)
    assert co.save_override("AgentConfig.arbiter_timeout_sec", 3.5)
    co.clear_cache()
    assert AgentConfig().arbiter_timeout_sec == pytest.approx(3.5)
    co._delete_all()
    co.clear_cache()


def test_a_reply_that_arrives_within_the_timeout_is_used():
    """4.2 秒で返る調停（「黙って」の実測に近い）を、時間切れにしない。時間切れを見るのは軽量LLM の文章。"""
    writer = writer_says({"text": "わかった"}, delay=0.05)
    decision = asyncio.run(
        decide(jev=jev_says("light"), writer=writer, utterance="黙って", timeout=0.5)
    )
    assert decision.branch == "light"


def test_a_reply_that_exceeds_the_timeout_falls_back_to_full():
    """届かなければフルへ倒す（従来どおり）。"""
    writer = writer_says({"text": "間に合わない"}, delay=0.3)
    decision = asyncio.run(
        decide(jev=jev_says("light"), writer=writer, utterance="黙って", timeout=0.05)
    )
    assert decision.branch == "full"
