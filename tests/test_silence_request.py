"""「黙っていて」と頼まれたら、その人が居るあいだ黙る。

- **気づくのは軽量LLM**（調停）。言い方は「黙って」「うるさい」「あとにして」「いま集中
  したい」と無数にあり、文字列の一覧は必ず漏れる。ツールを渡してフルLLM に呼ばせる形も
  取らない（調停は毎反復の頭で必ず通るので、そこで判断すれば発話の出口すべてに効く）。
- **止めるのは発話すべて**。自発だけでなく、話しかけられても話さない。
- **解けるのは退室と時間**。時間は Config（既定60分）。
- **黙っているあいだの言葉は捨てず `pending_speech` へ溜める**。「聞きたくない」ではなく
  「いまは」なので、解けたときに配る。
"""

from __future__ import annotations

import time

from familiar_agent.silence_state import (
    SilenceRequest,
    clear_silence,
    is_silenced,
)


def test_silenced_while_the_asker_is_present():
    req = SilenceRequest(person="パパ", until=time.time() + 600)
    assert is_silenced(req, present={"パパ"}, now=time.time()) is True


def test_not_silenced_when_the_asker_has_left():
    # 退室で解ける。
    req = SilenceRequest(person="パパ", until=time.time() + 600)
    assert is_silenced(req, present={"たいきくん"}, now=time.time()) is False


def test_not_silenced_after_the_time_runs_out():
    req = SilenceRequest(person="パパ", until=time.time() - 1)
    assert is_silenced(req, present={"パパ"}, now=time.time()) is False


def test_no_request_means_no_silence():
    assert is_silenced(None, present={"パパ"}, now=time.time()) is False


def test_arbiter_can_flag_a_silence_request():
    # 気づくのは軽量LLM。言い方は無数にあるので、文字列の一覧を持たない。
    from familiar_agent.loop.arbiter import ARBITER_PROMPT, Decision, arbitrate
    import asyncio
    from unittest.mock import AsyncMock

    assert "silence" in ARBITER_PROMPT
    b = AsyncMock()
    b.complete = AsyncMock(return_value='{"branch":"light","text":"わかった","silence":true}')
    d: Decision = asyncio.run(arbitrate(b, utterance="ちょっと静かにして", workspace_ctx=""))
    assert d.silence is True


def test_silence_blocks_speech_even_when_spoken_to():
    # 止めるのは発話すべて。自発だけでなく、話しかけられても話さない。
    import time as _time
    from unittest.mock import MagicMock

    from familiar_agent.loop.event_loop import InformationProcessing

    a = MagicMock()
    a._pmm.presence_status = MagicMock(
        return_value=[{"name": "パパ", "is_speaker": True, "confidence": 1.0}]
    )
    a._social_presence_permission = MagicMock(return_value=1.0)   # 相手は居る
    a._in_quiet_hours = MagicMock(return_value=False)             # 静穏時間でもない
    ip = InformationProcessing(a)
    req = SilenceRequest(person="パパ", until=_time.time() + 600)
    import familiar_agent.silence_state as ss

    original, ss.load_silence = ss.load_silence, lambda: req
    try:
        assert ip._delivery_block_reason() == "黙っているよう頼まれている"
    finally:
        ss.load_silence = original


def test_default_duration_is_sixty_minutes():
    from familiar_agent.config import AgentConfig

    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {}, clear=True):
        assert AgentConfig().silence_minutes == 60
