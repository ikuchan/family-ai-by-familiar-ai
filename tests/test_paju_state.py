"""環-aw 第 1 段 項目 1：パジュの状態図（頭と耳）の骨組みと、捨てた事象のログ（`設計方針_パジュの状態` v0.3 決定ス、ユ）。

頭は何もしていない、考えている（子 5 つ）、答えを言っている。耳は聞いている、考えているあいだ聞かない。耳は頭に
入るときの動作ではなく事象で動く（答えのあと考えているへ戻っても、聞いたまま）。
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from familiar_agent.loop.paju_state import PajuState


def _run(*events: str) -> PajuState:
    async def go():
        sm = PajuState()
        await sm.start()
        for e in events:
            await sm.fire(e)
        return sm

    return asyncio.run(go())


def _ids(sm: PajuState) -> set[str]:
    return {s.id for s in sm.configuration}


def test_the_chart_runs_on_the_async_engine():
    """**番人**：非同期の動作が 1 つも無いとライブラリは同期で動き、`start` を待てずに落ちる。"""
    from statemachine.engines.async_ import AsyncEngine

    assert isinstance(PajuState()._engine, AsyncEngine)


def test_before_start_it_counts_as_hearing():
    """活性化の前（状態が空）は、聞いているとみなす。"""
    sm = PajuState()
    assert not sm.configuration
    assert not sm.is_deaf()


def test_it_starts_idle_and_hearing():
    sm = _run()
    assert {"idle", "hearing"} <= _ids(sm)
    assert not sm.is_deaf()


def test_opening_a_request_thinks_and_stops_hearing():
    sm = _run("request_opened")
    assert {"thinking", "recalling", "deaf"} <= _ids(sm)
    assert sm.is_deaf()


@pytest.mark.parametrize(
    "event, child",
    [
        ("recall_began", "recalling"),
        ("decide_began", "deciding"),
        ("wait_began", "awaiting"),
        ("check_began", "checking"),
        ("note_began", "noting"),
    ],
)
def test_each_child_is_entered_by_its_event(event, child):
    sm = _run("request_opened", "decide_began", event)
    assert {"thinking", child} <= _ids(sm)


def test_speaking_is_a_sibling_and_the_ear_stays_deaf():
    """答えを言っているは考えているの兄弟。言っているあいだ、耳は聞かないまま（決定モ）。"""
    sm = _run("request_opened", "check_began", "speech_began")
    ids = _ids(sm)
    assert "answering" in ids and "thinking" not in ids
    assert sm.is_deaf()


def test_after_speaking_it_notes_and_hears():
    """**事象で動かす理由**：答えのあと考えている（書き留めている）へ戻っても、耳は聞く（決定モ）。"""
    sm = _run("request_opened", "speech_began", "speech_ended")
    assert {"thinking", "noting", "hearing"} <= _ids(sm)
    assert not sm.is_deaf()


def test_closing_returns_to_idle_and_hearing():
    for events in (
        ("request_opened", "request_closed"),
        ("request_opened", "speech_began", "request_closed"),
        ("request_opened", "speech_began", "speech_ended", "request_closed"),
    ):
        sm = _run(*events)
        assert {"idle", "hearing"} <= _ids(sm), events


def test_the_limit_makes_the_ear_hear():
    """上限に届いたら耳は聞く側へ戻る（決定ミ）。頭は時間切れの答えを言ってから閉じる。"""
    sm = _run("request_opened", "wait_began", "limit_reached")
    assert not sm.is_deaf()


def test_an_event_without_a_transition_is_logged_and_changes_nothing(caplog):
    """捨てた事象は、名前と各領域の状態を 1 行残す（決定ユ）。"""
    with caplog.at_level(logging.INFO, logger="familiar_agent.loop.paju_state"):
        sm = _run("speech_ended")
    assert {"idle", "hearing"} <= _ids(sm)
    lines = [r.getMessage() for r in caplog.records if "捨てた" in r.getMessage()]
    assert lines == [
        "状態図 speech_ended を受ける移りが無いので捨てた（頭=何もしていない 耳=聞いている）"
    ]


def test_an_event_with_a_transition_is_not_logged(caplog):
    """**反証の側**：移りのある事象では、捨てたログを書かない。"""
    with caplog.at_level(logging.INFO, logger="familiar_agent.loop.paju_state"):
        _run("request_opened", "decide_began")
    assert not [r for r in caplog.records if "捨てた" in r.getMessage()]


def test_the_description_names_the_child():
    sm = _run("request_opened", "decide_began")
    assert sm.describe() == "頭=考えている/決めている 耳=聞かない"


def test_speaking_again_after_hearing_stops_hearing():
    """言い終えて聞いたあと、同じ求めでもう一度言うあいだは、また聞かない（決定モ。途中の確認の問いのあとの答え）。"""
    sm = _run("request_opened", "speech_began", "speech_ended", "decide_began", "speech_began")
    assert {"answering", "deaf"} <= _ids(sm)


def test_entering_a_thinking_child_calls_the_hook():
    """考えているの子に入るたびに、子の名前で口が呼ばれる（電子音を鳴らす・決定テ）。耳と頭の外の状態では呼ばれない。"""
    entered: list[str] = []

    async def go():
        sm = PajuState(on_child=entered.append)
        await sm.start()
        for e in ("request_opened", "decide_began", "wait_began", "speech_began", "speech_ended"):
            await sm.fire(e)
        await sm.fire("request_closed")

    asyncio.run(go())
    assert entered == ["recalling", "deciding", "awaiting", "noting"]


def test_a_failing_hook_does_not_break_the_transition(caplog):
    """入るときの動作（電子音を鳴らし始める）が例外を出しても、移りは最後まで済み、WARNING に残る。

    ライブラリは動作の例外を内部の事象に変えて DEBUG にしか残さないので、黙って消えないようにする。
    """

    def broken(child):
        raise RuntimeError("鳴らし始められない")

    sm = PajuState(on_child=broken)

    async def go():
        await sm.start()
        await sm.fire("request_opened")
        return sm.head_state(), sm.is_deaf()

    with caplog.at_level(logging.WARNING):
        head, deaf = asyncio.run(go())
    assert head == "thinking" and deaf
    assert any("入るときの動作" in r.getMessage() for r in caplog.records)


def test_entering_idle_calls_the_idle_hook():
    """頭が何もしていないに入るたびに呼ぶ（初めの状態に入るときも、求めを閉じたときも）。"""
    idles: list = []
    sm = PajuState(on_idle=lambda: idles.append(sm.head_state()))

    async def go():
        await sm.start()
        await sm.fire("request_opened")
        await sm.fire("decide_began")
        assert len(idles) == 1  # 反証の側：考えているの子の移りでは呼ばない
        await sm.fire("request_closed")

    asyncio.run(go())
    assert len(idles) == 2
