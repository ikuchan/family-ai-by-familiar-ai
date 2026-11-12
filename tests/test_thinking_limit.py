"""考えているの上限（環-aw 第 1 段 項目 7、8・`設計方針_パジュの状態` v0.4 決定ミ、マ）。

求めを開いてから考えているの合計が上限（既定 3 分）に届いたら、状態図が `limit_reached` を送り、ループへ知らせる
（`on_limit`）。答えを言い始めたら、その求めではもう数えない（項目 8・2026-11-10 本人の決定）。何もしていないに
入ったら数えるのをやめ、次の求めは初めから数える。残り時間は状態図のファイルの中だけで持つ。主LLM を 1 回呼ぶ待ちは
60 秒で切る（決定マ）。
"""

from __future__ import annotations

import asyncio
import logging
import os
from unittest.mock import patch

from familiar_agent.loop.paju_state import PajuState

LOG = "familiar_agent.loop.paju_state"


def _limit_events(caplog) -> int:
    """考えているの上限に届いた数（数える仕事が残す INFO の行。耳が聞いていて送らなかった回も数える）。"""
    return sum("考えているが上限" in r.getMessage() for r in caplog.records)


def _sent(caplog) -> int:
    """状態図が `limit_reached` を受けた数（移ったときの DEBUG と、捨てたときの INFO の両方）。"""
    return sum("limit_reached" in r.getMessage() for r in caplog.records)


def _play(caplog, limit, steps):
    """上限 `limit` 秒の状態図に、事象（文字列）と待ち（秒）を順に与え、終わりの状態図を返す。"""

    async def go():
        sm = PajuState(thinking_limit_sec=limit)
        await sm.start()
        for s in steps:
            if isinstance(s, str):
                await sm.fire(s)
            else:
                await asyncio.sleep(s)
        return sm

    with caplog.at_level(logging.DEBUG, logger=LOG):
        return asyncio.run(go())


def test_thinking_too_long_reaches_the_limit_and_hears_again(caplog):
    sm = _play(caplog, 0.1, ["request_opened", 0.4])
    assert _limit_events(caplog) == 1 and _sent(caplog) == 1
    assert not sm.is_deaf()  # 耳は聞く側へ戻る
    assert sm.head_state() == "thinking"  # 頭は変えない（求めを閉じるのは項目 8）


def test_before_the_limit_nothing_happens(caplog):
    """**反証の側**：上限より前は、耳は聞かないまま。"""
    sm = _play(caplog, 0.5, ["request_opened", 0.1])
    assert _limit_events(caplog) == 0
    assert sm.is_deaf()


def test_time_spent_answering_is_not_counted(caplog):
    """答えを言っているあいだは数えない。"""
    _play(caplog, 0.3, ["request_opened", 0.1, "speech_began", 0.5])
    assert _limit_events(caplog) == 0  # 数えていれば 0.3 秒で届いている


def test_after_the_answer_it_is_no_longer_counted(caplog):
    """**反証の側**：話し終えて書き留めに戻っても、その求めではもう数えない（続けて数えれば残りの 0.2 秒で届く）。"""
    _play(caplog, 0.3, ["request_opened", 0.1, "speech_began", 0.05, "speech_ended", 0.5])
    assert _limit_events(caplog) == 0


def test_the_next_request_after_an_answer_counts_again(caplog):
    """数えるのをやめるのは、答えた求めだけ。次の求めは初めから数える。"""
    _play(
        caplog,
        0.1,
        ["request_opened", "speech_began", "speech_ended", "request_closed", "request_opened", 0.3],
    )
    assert _limit_events(caplog) == 1


def test_closing_the_request_stops_the_count(caplog):
    _play(caplog, 0.15, ["request_opened", 0.05, "request_closed", 0.4])
    assert _limit_events(caplog) == 0


def test_the_next_request_counts_from_the_start(caplog):
    """前の求めで使った時間を、次の求めへ持ち越さない。"""
    _play(caplog, 0.3, ["request_opened", 0.2, "request_closed", "request_opened", 0.15])
    assert _limit_events(caplog) == 0
    caplog.clear()
    _play(caplog, 0.3, ["request_opened", 0.2, "request_closed", "request_opened", 0.5])
    assert _limit_events(caplog) == 1


def test_a_zero_limit_is_reached_at_once(caplog):
    """上限 0 は「数えない」ではなく「すぐ届く」。"""
    _play(caplog, 0.0, ["request_opened", 0.05])
    assert _limit_events(caplog) == 1


def test_it_is_sent_once_per_request(caplog):
    """**反証の側**：届いたあと答えて戻っても、同じ求めでもう一度は送らない。"""
    _play(caplog, 0.1, ["request_opened", 0.3, "speech_began", "speech_ended", 0.3])
    assert _limit_events(caplog) == 1


def test_answering_right_at_the_limit_ends_the_count(caplog):
    """上限の刻みに答え始めても（数える仕事はまだ動いていない）、その求めではもう届かない。"""

    async def go():
        sm = PajuState(thinking_limit_sec=10.0)
        await sm.start()
        await sm.fire("request_opened")
        sm._limit_since -= 11.0  # 上限を過ぎたが、数える仕事はまだ動いていない（時計は止めない：loop の時計も止まる）
        await sm.fire("speech_began")
        left = sm._limit_left
        await sm.fire("speech_ended")
        await asyncio.sleep(0.05)
        return left

    with caplog.at_level(logging.DEBUG, logger=LOG):
        assert asyncio.run(go()) is None
    assert _limit_events(caplog) == 0


def test_no_limit_event_into_a_hearing_ear(caplog):
    """電子音を鳴らせず耳がもう聞いていれば、`limit_reached` を送らない（移りの無い事象は、列に待つほかの
    事象もライブラリに退けさせる）。届いたことは 1 行残す。"""
    _play(caplog, 0.1, ["request_opened", "limit_reached", 0.3])
    assert _limit_events(caplog) == 1
    assert _sent(caplog) == 1  # 電子音の失敗の代わりに送った 1 回だけ


def _told(limit, steps):
    """`on_limit` が呼ばれた回数。"""
    calls: list = []

    async def go():
        sm = PajuState(thinking_limit_sec=limit, on_limit=lambda: calls.append(1))
        await sm.start()
        for s in steps:
            if isinstance(s, str):
                await sm.fire(s)
            else:
                await asyncio.sleep(s)

    asyncio.run(go())
    return len(calls)


def test_reaching_the_limit_tells_the_loop():
    assert _told(0.1, ["request_opened", 0.3]) == 1


def test_the_loop_is_told_even_when_the_ear_already_hears():
    """耳が聞いているかは、事象を送るかだけを決める。考えすぎたことは知らせる。"""
    assert _told(0.1, ["request_opened", "limit_reached", 0.3]) == 1


def test_the_loop_is_not_told_without_reaching_the_limit():
    """**反証の側**：電子音の失敗で送る `limit_reached`・上限より前・答えたあとは、知らせない。"""
    assert _told(10.0, ["request_opened", "limit_reached", 0.1]) == 0
    assert _told(0.5, ["request_opened", 0.1]) == 0
    assert _told(0.2, ["request_opened", "speech_began", "speech_ended", 0.4]) == 0


def test_the_count_runs_outside_the_charts_processing():
    """数える仕事は、状態図の処理の段の文脈を受け継がない（受け継ぐと、送る事象に待つ口が付かず列に残る）。"""
    from statemachine.engines.base import _running_engine

    seen: list = []

    async def go():
        sm = PajuState(thinking_limit_sec=10.0)

        async def probe(seconds, limit):
            seen.append(_running_engine.get())

        sm._wait_limit = probe  # type: ignore[method-assign]
        await sm.start()
        await sm.fire("request_opened")
        await asyncio.sleep(0.05)

    asyncio.run(go())
    assert seen == [None]


def test_spawn_outside_does_not_carry_context_variables():
    import contextvars

    from familiar_agent.core.aio import spawn_outside

    var: "contextvars.ContextVar[str]" = contextvars.ContextVar("v", default="外")

    async def read():
        return var.get()

    async def go():
        var.set("中")
        inside = await asyncio.get_running_loop().create_task(read())
        outside = await spawn_outside(read())
        return inside, outside

    assert asyncio.run(go()) == ("中", "外")  # 反証の側：ふつうに立てれば受け継ぐ


def test_closing_the_loop_stops_the_count(caplog):
    from tests.test_beeps_while_waiting import _ip

    ip, _ = _ip()
    ip._agent.config.thinking_limit_sec = 0.15
    ip._paju_state = None  # 状態図はループを作った時点の上限で作られるので、変えた値で作り直す

    async def go():
        await ip._to_state("request_opened")
        ip._stop_beeps()
        await ip.close()
        # 閉じたあとに考えているへ入り直しても、数え始めない（見直しの指摘）
        await ip._to_state("request_closed")
        await ip._to_state("request_opened")
        ip._stop_beeps()
        await asyncio.sleep(0.4)

    with caplog.at_level(logging.DEBUG, logger=LOG):
        asyncio.run(go())
    assert _limit_events(caplog) == 0


def test_without_a_limit_it_never_counts(caplog):
    """上限を渡さなければ数えない（状態図だけを試す道）。"""
    _play(caplog, None, ["request_opened", 0.1])
    assert _limit_events(caplog) == 0


def test_the_loop_gives_the_chart_the_configured_limit(caplog):
    from tests.test_beeps_while_waiting import _ip

    ip, _ = _ip()
    ip._agent.config.thinking_limit_sec = 0.1
    ip._paju_state = None  # 状態図はループを作った時点の上限で作られるので、変えた値で作り直す

    async def go():
        await ip._to_state("request_opened")
        await asyncio.sleep(0.4)
        deaf = ip._paju().is_deaf()
        ip._stop_beeps()
        await ip.close()
        return deaf

    with caplog.at_level(logging.DEBUG, logger=LOG):
        assert asyncio.run(go()) is False
    assert _limit_events(caplog) == 1


def test_the_limits_come_from_config():
    from familiar_agent.config import AgentConfig

    with patch.dict(os.environ, {}, clear=True):
        c = AgentConfig()  # 設定を丸ごと表示させない（鍵が出る）
        thinking, main_llm = c.thinking_limit_sec, c.main_llm_timeout_sec
    assert thinking == 180.0  # 決定ミ：考えているの合計 3 分
    assert main_llm == 60.0  # 決定マ：主LLM 1 回の待ち


def test_the_beeps_run_outside_the_charts_processing():
    """電子音の仕事も、状態図に入るときの動作から立つので、処理の段の文脈を受け継がない（鳴らせないときに送る事象のため）。"""
    from statemachine.engines.base import _running_engine

    from tests.test_beeps_while_waiting import _ip

    ip, _ = _ip()
    seen: list = []

    async def probe(gen):
        seen.append(_running_engine.get())

    ip._beep_while_waiting = probe  # type: ignore[method-assign]

    async def go():
        await ip._to_state("request_opened")
        await asyncio.sleep(0.05)
        ip._stop_beeps()
        await ip.close()

    asyncio.run(go())
    assert seen == [None]
