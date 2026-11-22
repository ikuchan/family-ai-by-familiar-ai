"""記-ae：窓の外で捨てた声を、数えるだけ残す（2026-11-22）。

窓の外の声は捨てて記録しない（出-as 段 3・本人の決まり）。家族どうしの話を about の材料にするか（決まりを変えるか）を本人に
聞くために、捨てた声のうち「家族の声と分かったか」「ほかの家族の呼び名が出たか」だけを 1 行に数える。言葉も名前も出さず、
何も記録しない（決まりは変えない）。
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

from tests.test_voice_picks_the_speaker import PAPA, SABURO, _ip

LOGGER = "familiar_agent.loop.event_loop"


def _push(ip, text, voice):
    async def go():
        ip._swallow_if_unheard = AsyncMock(return_value=True)  # type: ignore[method-assign]
        out = await ip.push_utterance(text, source="voice", voice=voice)
        await ip.close()
        return out

    return asyncio.run(go())


def _counted(caplog) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("窓の外の声を数えた")
    ]


def test_a_family_voice_naming_another_member_is_counted(caplog):
    ip, a, s = _ip()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _push(ip, "いちろう、最近忙しそうだね", SABURO)  # さぶろうの声でパパの呼び名
    (line,) = _counted(caplog)
    assert line == "窓の外の声を数えた（記-ae）：家族の声 はい・ほかの家族の呼び名 はい"
    assert s.added == []  # 何も記録しない


def test_the_speakers_own_name_is_not_another_member(caplog):
    ip, a, s = _ip()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _push(ip, "いちろうはね、今日は早く寝る", PAPA)  # パパの声で自分の呼び名
    (line,) = _counted(caplog)
    assert line.endswith("家族の声 はい・ほかの家族の呼び名 いいえ")


def test_the_line_has_no_words_and_no_names(caplog):
    ip, a, s = _ip()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _push(ip, "いちろう、最近忙しそうだね", SABURO)
    (line,) = _counted(caplog)
    assert "いちろう" not in line and "忙しそう" not in line and "さぶろう" not in line
