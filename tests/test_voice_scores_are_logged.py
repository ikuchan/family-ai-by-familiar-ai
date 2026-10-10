"""声が届くたびに、全員の似かたを INFO に残す（知-as 段 1・2026-10-10・本人の決定）。

二人が交互に話しても話者が切り替わらない（実機 10:56〜11:00）。書き起こした声 65 回のうち 41 回は窓の外として捨てられ、
照らしていない。通した声でも似かたは DEBUG にしか残らず、何が効いているか（窓の外・付け替えの基準・登録が 1 回分・指定が
60 秒で切れる）を点数で確かめられない。まず測る：声付きの入力 1 回につき、全員の点数と判定を 1 行残す。捨てた声は点数と
「通していたら」の答えだけ残し、話者・今日の声には触らない。本文は出さない。
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

import numpy as np

from tests.test_voice_picks_the_speaker import PAPA, TAIKI, _ip

LOGGER = "familiar_agent.loop.event_loop"
SAID = "ひみつのことば"


def _lines(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO and r.getMessage().startswith("声の似かた")
    ]


def test_a_passed_voice_logs_every_score_and_the_verdict(caplog):
    ip, a, s = _ip()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        asyncio.run(ip._match_voice(TAIKI))
    lines = _lines(caplog)
    assert len(lines) == 1
    line = lines[0]
    assert "通した" in line and "switch" in line
    assert "たいき 1.00" in line and "パパ 0.00" in line


def _push(ip, voice):
    async def go():
        ip._swallow_if_unheard = AsyncMock(return_value=True)  # type: ignore[method-assign]
        out = await ip.push_utterance(SAID, source="voice", voice=voice)
        await ip.close()
        return out

    return asyncio.run(go())


def test_a_swallowed_voice_logs_scores_but_changes_nothing(caplog):
    ip, a, s = _ip()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _push(ip, TAIKI)
    lines = _lines(caplog)
    assert len(lines) == 1
    assert "捨てた" in lines[0] and "通していたら switch" in lines[0]
    assert "たいき 1.00" in lines[0]
    a._persons.set_active.assert_not_called()
    a._persons.reset_to_default.assert_not_called()
    assert s.added == []


def test_no_line_without_a_voice(caplog):
    ip, a, s = _ip()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _push(ip, None)
        asyncio.run(ip._match_voice(None))
    assert _lines(caplog) == []


def test_the_words_are_not_in_the_line(caplog):
    ip, a, s = _ip()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _push(ip, PAPA)
    assert _lines(caplog) and all(SAID not in line for line in _lines(caplog))


def test_the_unknown_check_uses_the_same_scores():
    """`_voice_unknown` も同じ点数（`_voice_scores`）で決める。"""
    ip, a, s = _ip()
    assert ip._voice_unknown(TAIKI) is False
    assert ip._voice_unknown(np.asarray([-1.0, -1.0], dtype=np.float32)) is True
    assert ip._voice_scores(TAIKI) == {"papa": 0.0, "taiki": 1.0}
