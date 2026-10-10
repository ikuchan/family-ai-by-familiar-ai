"""`/voice` の受付は、話し始めた時刻で見る（知-au・2026-10-10・実機 11:53・本人の決定）。

`/voice こうき` の受付（10 秒）の間にこうきが話し始めたのに、書き起こしに 6.77 秒かかり、書き起こしが届いた時刻で
締め切りを見たので「受付が過ぎた」になった。声の入力には話し始めた時刻（`InputText.at`・`time.monotonic()`）が付いて
いるので、受付を開いた時刻から締め切りまでの間に話し始めた声なら、書き起こしが遅れても登録に使う。受付を開く前から
話していた声は使わない（`/voice` を打つ前の会話を登録しない）。
"""

from __future__ import annotations

import asyncio

from familiar_agent.agent import EmbodiedAgent as Agent
from familiar_agent.core.wake_window import KeyText, VoiceText
from tests.test_voice_enrollment import VOICE, _agent

OPEN = 1000.0


def _open(a):
    asyncio.run(Agent.run(a, KeyText("/voice パパ", at=OPEN)))


def test_a_voice_started_inside_the_window_is_used_even_if_transcribed_late():
    clock = {"t": OPEN}
    a, store = _agent(clock)
    _open(a)
    clock["t"] = OPEN + 14.0  # 書き起こしが届いたのは締め切りの後（実機：6.77 秒かかった）
    out = asyncio.run(Agent.run(a, VoiceText("こんにちは、パジュ", at=OPEN + 0.1, voice=VOICE)))
    assert "声を登録しました" in out
    assert store.add.call_count == 2


def test_a_voice_started_before_the_command_is_not_used():
    a, store = _agent({"t": OPEN})
    _open(a)
    asyncio.run(Agent.run(a, VoiceText("さっきの話", at=OPEN - 2.0, voice=VOICE)))
    store.add.assert_not_called()


def test_a_voice_started_after_the_window_is_not_used():
    a, store = _agent({"t": OPEN})
    _open(a)
    asyncio.run(Agent.run(a, VoiceText("こんにちは", at=OPEN + 11.0, voice=VOICE)))
    store.add.assert_not_called()
    assert a._voice_enroll is None
