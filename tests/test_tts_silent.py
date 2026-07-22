"""TTS_OUTPUT=silent で speaker 出力を止める（実機テスト用）。

api_key があってもスピーカーへ出さず、合成 API も叩かない（早期 return）。
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_say_silent_skips_speaker_and_api():
    from familiar_agent.tools.tts import TTSTool

    t = TTSTool("fake-key", "fake-voice", output="silent")
    result = await t.say("こんにちは、テストです")
    assert result.startswith("Said (silent):")
