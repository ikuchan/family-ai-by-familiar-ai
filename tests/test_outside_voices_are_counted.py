"""記-ae：窓の外で捨てた声を数え、家族の声でほかの家族の呼び名が出た言葉だけを記録する（2026-11-22・11-23）。

2026-11-23 本人の決定：窓の外の声は捨てて記録しない決まり（出-as 段 3）を、狭く変える。声で家族の誰かと分かり、ほかの家族の
呼び名が出た言葉だけを、向き「聞こえた」で、話した人の記録として O に残す（パジュは何もしない）。それ以外はいまどおり捨てる。

以下は 2026-11-22 の数えるだけの段の説明：

窓の外の声は捨てて記録しない（出-as 段 3・本人の決まり）。家族どうしの話を about の材料にするか（決まりを変えるか）を本人に
聞くために、捨てた声のうち「家族の声と分かったか」「ほかの家族の呼び名が出たか」だけを 1 行に数える。言葉も名前も出さず、
何も記録しない（決まりは変えない）。
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from unittest.mock import AsyncMock

from tests.test_voice_picks_the_speaker import PAPA, SABURO, _ip

LOGGER = "familiar_agent.loop.event_loop"


def _push(ip, text, voice):
    """捨てる理由は問わず、捨てた声として通す（数える行を見る試験）。"""

    async def go():
        ip._agent._oif.write = AsyncMock(return_value="obs-1")
        ip._swallow_if_unheard = AsyncMock(return_value=True)  # type: ignore[method-assign]
        return await _run(ip, text, voice)

    return asyncio.run(go())


def _overhear(ip, text, voice):
    """本物の入口で、窓の外の声として捨てさせる（名前なし・窓は閉じている）。裏の書き込みを待つ。"""

    async def go():
        ip._agent._oif.write = AsyncMock(return_value="obs-1")
        await ip.push_utterance(text, source="voice", voice=voice)
        if ip._background_tasks:
            await asyncio.gather(*ip._background_tasks)
        await ip.close()

    asyncio.run(go())


async def _run(ip, text, voice):
    out = await ip.push_utterance(text, source="voice", voice=voice)
    await ip.close()
    return out


def _written(ip):
    return [
        (c.args[0].direction, c.args[0].content, c.kwargs.get("writer_id"))
        for c in ip._agent._oif.write.await_args_list
    ]


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


@pytest.mark.real_window
def test_a_family_voice_naming_another_member_outside_the_window_is_recorded_as_heard():
    ip, a, s = _ip()
    _overhear(ip, "いちろう、最近忙しそうだね", SABURO)
    assert _written(ip) == [("聞こえた", "いちろう、最近忙しそうだね", "saburo")]


@pytest.mark.real_window
def test_without_another_members_name_nothing_is_recorded():
    ip, a, s = _ip()
    _overhear(ip, "いちろうはね、今日は早く寝る", PAPA)  # 自分の呼び名だけ
    _overhear(ip, "今日は雨だね", SABURO)  # 呼び名なし
    assert _written(ip) == []


@pytest.mark.real_window
def test_an_unrecognised_voice_is_not_recorded():
    import numpy as np

    ip, a, s = _ip()
    _overhear(ip, "いちろう、最近忙しそうだね", np.asarray([0.7, -0.7], dtype=np.float32))
    assert _written(ip) == []


def test_a_voice_dropped_for_another_reason_is_not_recorded():
    """窓の外・家族どうし以外の理由で捨てた声（考えている最中など）は、パジュ宛てかもしれないので残さない。"""
    ip, a, s = _ip()
    _push(ip, "いちろう、最近忙しそうだね", SABURO)  # 捨てる理由を問わない口（数えるだけ）
    assert _written(ip) == []


def test_a_heard_record_reads_as_not_said_to_paju():
    from familiar_agent.tools.memory import subject_line

    assert subject_line("聞こえた", "ママ") == "ママが話していた（パジュ宛てではない）"
