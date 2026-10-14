"""確認の問いを出しているあいだと確認待ちのあいだは、つなぎを出さない（出-ci・2026-10-14・本人の決定ア）。

10/14 10:13 の「3分のタイマーをかけて」で、確認の問い（「3 分のタイマーね。…いい？」）を読んでいる最中に、待ちの見張りが
話し始めから 8 秒を超えたとしてつなぎを頼み、問いの直後に「はい、ただいま設定いたしますので少々お待ちください。」が流れた。
問いは道具の呼び出しの中で読むので、そのあいだ `set_timer` は待たせているものに数えられていた。
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from tests.test_filler_gap_after_sound import _counts, _ip
from tests.test_fillers_are_written_in_the_background import _ip as _writing_ip


def _late(ip):
    def setup():
        import time

        now = time.monotonic()
        ip._req.heard_at = now - 10.0  # とうに 8 秒を超えている

    return setup


def test_no_filler_is_asked_for_while_confirming():
    ip = _ip()
    ip._agent.confirm_alive = lambda: True
    assert _counts(ip, setup=_late(ip), checks=(0.15,)) == [0]


def test_a_filler_is_asked_for_when_not_confirming():
    """**反証の側**：確認待ちでなければ、今までどおり頼む。"""
    ip = _ip()
    ip._agent.confirm_alive = lambda: False
    assert _counts(ip, setup=_late(ip), checks=(0.15,)) == [1]


def test_a_filler_written_before_the_question_is_not_said():
    """書いているあいだに確認待ちが置かれたら、書き上がったつなぎは言わない。"""
    ip, said = _writing_ip()
    ip._agent.confirm_alive = lambda: False

    async def go():
        await ip._say_waiting_filler("x", "", 1)
        ip._agent.confirm_alive = lambda: True  # 書いているあいだに問いを出し始めた
        await asyncio.gather(*ip._filler_jobs)
        await ip.close()

    asyncio.run(go())
    assert said == []
    assert ip._req.filler_pending is False


def test_a_written_filler_is_said_when_not_confirming():
    """**反証の側**：確認待ちでなければ、書き上がったつなぎを言う。"""
    ip, said = _writing_ip()
    ip._agent.confirm_alive = MagicMock(return_value=False)

    async def go():
        await ip._say_waiting_filler("x", "", 1)
        await asyncio.gather(*ip._filler_jobs)
        await ip.close()

    asyncio.run(go())
    assert said == ["もう少しね"]
