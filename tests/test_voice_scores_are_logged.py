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

from tests.test_voice_picks_the_speaker import PAPA, SABURO, _ip
from familiar_agent.core.redact import tag

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
        asyncio.run(ip._match_voice(SABURO))
    lines = _lines(caplog)
    assert len(lines) == 1
    line = lines[0]
    assert "通した" in line and "switch" in line
    # 名前は出さず、印にする（出-dm）
    assert f"{tag('さぶろう')} 1.00" in line and f"{tag('パパ')} 0.00" in line
    assert "さぶろう" not in line and "パパ" not in line


def _push(ip, voice):
    async def go():
        ip._swallow_if_unheard = AsyncMock(return_value=True)  # type: ignore[method-assign]
        out = await ip.push_utterance(SAID, source="voice", voice=voice)
        await ip.close()
        return out

    return asyncio.run(go())


def test_a_swallowed_voice_logs_scores_but_changes_nothing(caplog):
    """点数が窓の外の付け替えの基準に届かなければ、捨てた声は話者・今日の声に触らない。"""
    ip, a, s = _ip()
    a.config.recognition.voice_outside_switch_threshold = 1.01  # 1.00 の声でも届かない
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _push(ip, SABURO)
    lines = _lines(caplog)
    assert len(lines) == 1
    assert "捨てた" in lines[0] and "通していたら switch" in lines[0]
    assert f"{tag('さぶろう')} 1.00" in lines[0] and "さぶろう" not in lines[0]
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
    assert ip._voice_unknown(SABURO) is False
    assert ip._voice_unknown(np.asarray([-1.0, -1.0], dtype=np.float32)) is True
    assert ip._voice_scores(SABURO) == {"papa": 0.0, "saburo": 1.0}


def test_a_clearly_known_voice_outside_the_window_switches_the_speaker(caplog):
    """窓の外で捨てた声でも、点数が基準（0.55・仮）以上なら話者を付け替える（知-as 段 2・2026-11-21）。

    二人が交互に話しても、パジュの名前を呼ばない限り話者が変わらなかった。ログの 14,958 回の声のうち 97% は窓の外で、
    そのうち 3,073 回は通していたら付け替えていた。返事はしない（捨てた声のまま）。今日の声にも足さない。
    """
    ip, a, s = _ip()
    a._speaker_set_at = 0.0  # 指定はとうに切れている
    assert a.config.recognition.voice_outside_switch_threshold == 0.55
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = _push(ip, SABURO)
    assert out == ""  # 返事はしない
    a._persons.set_active.assert_called_once_with("さぶろう")
    assert a._speaker_set_at == 0.0  # 指定の時刻は新しくしない
    assert s.added == []  # 今日の声には足さない


def test_the_threshold_itself_switches_and_just_above_does_not():
    ip, a, s = _ip()
    a._speaker_set_at = 0.0
    a.config.recognition.voice_outside_switch_threshold = 1.0  # SABURO は 1.00
    _push(ip, SABURO)
    a._persons.set_active.assert_called_once()
    ip, a, s = _ip()
    a._speaker_set_at = 0.0
    a.config.recognition.voice_outside_switch_threshold = 1.0001
    _push(ip, SABURO)
    a._persons.set_active.assert_not_called()


def test_a_speaker_set_on_purpose_is_not_taken_over_from_outside():
    """`/speaker`・名乗り・写真・門を通った声で付けた話者は、指定が生きているあいだ（60 秒）窓の外の声で替えない。"""
    import time

    ip, a, s = _ip()
    a._speaker_set_at = time.time()
    _push(ip, SABURO)
    a._persons.set_active.assert_not_called()


def test_an_unknown_voice_outside_does_not_reset_the_speaker():
    import numpy as np

    ip, a, s = _ip()
    a._speaker_set_at = 0.0
    _push(ip, np.asarray([0.7071, -0.7071], dtype=np.float32))  # 誰にも当たらない
    a._persons.reset_to_default.assert_not_called()
    a._persons.set_active.assert_not_called()
