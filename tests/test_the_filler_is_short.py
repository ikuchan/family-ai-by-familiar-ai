"""つなぎは 15 字〔仮〕まで。超えたら作り置きの短い言葉に替える（出-cj・2026-10-14・本人の決定ア・15 字）。

10/14 10:13 のつなぎ「はい、ただいま設定いたしますので少々お待ちください。」は 26 字で、声で 4.3 秒かかった。軽量LLM には
「短い一言」とだけ頼んでいて、字数の決まりも機械の検めも無かった（集めた 42 件の中央は 14 字・長いものは 29 字）。
"""

from __future__ import annotations

import asyncio

from familiar_agent.core import filler_length as fl
from tests.test_fillers_are_written_in_the_background import _ip as _writing_ip
from tests.test_the_wait_filler_needs_no_judgement import _decided, _progress
from tests._arbiter_fakes import writer_says

LONG = "はい、ただいま設定いたしますので少々お待ちください。"


def test_a_short_filler_is_kept():
    """**反証の側**：15 字以内はそのまま。"""
    assert (
        fl.fit("はい、少々お待ちくださいね。", said=[], cue_words=[])
        == "はい、少々お待ちくださいね。"
    )
    assert len("少し待っててね、調べてるよ。。") == 15
    assert (
        fl.fit("少し待っててね、調べてるよ。。", said=[], cue_words=[])
        == "少し待っててね、調べてるよ。。"
    )


def test_a_long_filler_becomes_a_short_one():
    assert len(LONG) == 26
    assert fl.fit(LONG, said=[], cue_words=[]) == "ちょっと待ってね"


def test_the_short_one_avoids_the_cue_word():
    """合図で「待ってね」が鳴っていれば、それを含む「ちょっと待ってね」は使わない。"""
    assert fl.fit(LONG, said=[], cue_words=["待ってね"]) == "少し待っててね"


def test_the_short_one_avoids_what_was_said():
    assert fl.fit(LONG, said=["ちょっと待ってね"], cue_words=[]) == "少し待っててね"


def test_nothing_is_left_when_both_were_used():
    assert fl.fit(LONG, said=["ちょっと待ってね", "少し待っててね"], cue_words=[]) == ""


def test_the_writer_is_asked_for_fifteen_chars():
    writer = writer_says({"filler": "もう少しです"})
    _progress("search_deferred", writer=writer)
    assert "15 字以内" in _decided(writer)


def test_a_long_written_filler_is_said_short():
    ip, said = _writing_ip()

    async def write(inp, what, avoid=()):
        return LONG

    from unittest.mock import MagicMock

    ip._arbiter = lambda: MagicMock(write_filler=write)  # type: ignore[method-assign]

    async def go():
        await ip._say_waiting_filler("x", "", 1)
        await asyncio.gather(*ip._filler_jobs)
        await ip.close()

    asyncio.run(go())
    assert said == ["ちょっと待ってね"]
