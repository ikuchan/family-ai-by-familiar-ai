"""時間切れになった調停が、実際は何秒かかったのかを残す。

実機で「パジュ、一分黙って」だけが 2 秒の時間切れに掛かった（普通の会話は 0.93〜1.10 秒）。
ところが**時間切れの秒数しか分からない**ので、実際に 2.1 秒なのか 10 秒なのかが分からず、
時間切れの値を決められなかった。

応答は従来どおり 2 秒で倒す（体感を変えない）。**裏で待って、かかった秒数だけ残す。**
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from tests._arbiter_fakes import decide, jev_says, writer_says


def _light_written_in(delay: float):
    """Jev は light と決め、軽量LLM が返事を書くのに `delay` 秒かかる。"""
    return dict(jev=jev_says("light"), writer=writer_says({"text": "はい"}, delay=delay))


@pytest.mark.asyncio
async def test_a_slow_arbiter_still_falls_back_quickly():
    # 体感は変えない。倒す時刻は timeout のまま。
    started = asyncio.get_running_loop().time()
    d = await decide(**_light_written_in(1.0), utterance="黙って", timeout=0.05)
    assert asyncio.get_running_loop().time() - started < 0.5
    assert d.branch == "full"


@pytest.mark.asyncio
async def test_how_long_it_actually_took_is_recorded(caplog):
    """打ち切ったあとも裏で待ち、実際の秒数を残す。"""
    with caplog.at_level(logging.INFO, logger="familiar_agent.loop.arbiter"):
        await decide(**_light_written_in(0.2), utterance="黙って", timeout=0.05)
        await asyncio.sleep(0.4)  # 裏の完了を待つ
    assert any("遅れて返った" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_a_fast_arbiter_records_its_time_as_before(caplog):
    with caplog.at_level(logging.INFO, logger="familiar_agent.loop.arbiter"):
        d = await decide(**_light_written_in(0.0), utterance="やあ", timeout=2.0)
    assert d.branch == "light"
    assert any("調停 " in r.message for r in caplog.records)
