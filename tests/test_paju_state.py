"""環-aw 第 1 段 項目 1：パジュの状態図（頭と耳）の骨組みと、捨てた事象のログ（`設計方針_パジュの状態` v0.4 決定ス、ユ）。

頭は何もしていない、考えている（子 5 つ）、答えを言っている。耳は聞いている、考えているあいだ聞かない。耳は頭に
入るときの動作ではなく事象で動く（答えのあと考えているへ戻っても、聞いたまま）。
"""

from __future__ import annotations

import asyncio
import logging
import time

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
        "状態図 speech_ended を受ける移りが無いので捨てた（頭=何もしていない 耳=聞いている/窓=閉じている 口=話してよい）"
    ]


def test_an_event_with_a_transition_is_not_logged(caplog):
    """**反証の側**：移りのある事象では、捨てたログを書かない。"""
    with caplog.at_level(logging.INFO, logger="familiar_agent.loop.paju_state"):
        _run("request_opened", "decide_began")
    assert not [r for r in caplog.records if "捨てた" in r.getMessage()]


def test_the_description_names_the_child():
    sm = _run("request_opened", "decide_began")
    assert sm.describe() == "頭=考えている/決めている 耳=聞かない/窓=閉じている 口=話してよい"


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


# ---- 環-aw 第 2 段 項目 1：耳の窓と、口の黙るか（形だけ） ----

#: 1 つの領域の状態。どの領域も、同時に 1 つだけ取る。
_REGIONS = {
    "頭": {"idle", "thinking", "answering"},
    "考えているあいだ": {"hearing", "deaf"},
    "窓": {"shut", "open", "tentative"},
    "黙るか": {"may_speak", "silent"},
}


def _one_each(sm: PajuState) -> None:
    ids = _ids(sm)
    for name, states in _REGIONS.items():
        assert len(ids & states) == 1, (name, sorted(ids & states))


def test_it_starts_with_the_window_shut_and_free_to_speak():
    sm = _run()
    assert {"idle", "hearing", "shut", "may_speak"} <= _ids(sm)
    _one_each(sm)


def test_opening_a_request_leaves_the_window_and_the_mouth_alone():
    """耳の考えているあいだと窓は並行の領域。求めを開いても、窓は閉じたまま、口は話してよいのまま。"""
    sm = _run("request_opened")
    assert sm.is_deaf()
    assert {"deaf", "shut", "may_speak"} <= _ids(sm)


@pytest.mark.parametrize(
    "events, state",
    [
        (("window_opened",), "open"),
        (("window_opened", "window_lapsed"), "shut"),
        (("tentative_called",), "tentative"),
        (("tentative_called", "tentative_lapsed"), "shut"),
        (("tentative_called", "tentative_dropped"), "shut"),
        (("tentative_called", "window_opened"), "open"),
        (("silence_began",), "silent"),
        (("silence_began", "silence_ended"), "may_speak"),
    ],
)
def test_each_new_event_moves_its_region(events, state):
    sm = _run(*events)
    assert state in _ids(sm)
    _one_each(sm)


def test_re_entering_a_state_keeps_one_state_per_region():
    """**番人**：同じ状態へ戻る移りは、外へ出て入り直す形にする。`internal=True` の形では、並行の領域の中で
    ほかの領域の状態が二重になった（python-statemachine 3.2.2、2026-11-12 に作業用の場所で確かめた）。"""
    sm = _run(
        "request_opened",
        "window_opened",
        "window_opened",
        "silence_began",
        "silence_began",
        "window_lapsed",
        "tentative_called",
        "tentative_called",
    )
    _one_each(sm)
    assert {"thinking", "deaf", "tentative", "silent"} <= _ids(sm)


@pytest.mark.parametrize(
    "event", ["window_lapsed", "tentative_lapsed", "tentative_dropped", "silence_ended"]
)
def test_a_new_event_without_a_transition_is_dropped_and_changes_nothing(event, caplog):
    with caplog.at_level(logging.INFO, logger="familiar_agent.loop.paju_state"):
        sm = _run("request_opened", event)
    assert {"thinking", "deaf", "shut", "may_speak"} <= _ids(sm)
    _one_each(sm)
    assert [
        r
        for r in caplog.records
        if f"状態図 {event} を受ける移りが無いので捨てた" in r.getMessage()
    ]


def test_the_description_names_the_window_and_the_mouth():
    sm = _run("tentative_called", "silence_began")
    assert sm.describe() == "頭=何もしていない 耳=聞いている/窓=仮の呼びかけ 口=黙っている"


# ---- 環-aw 第 2 段 項目 2：窓 ----
# 窓の中だったかは、届いた時刻でデータ（`wake`）から決める。状態は、数える仕事が時間で移す（第 2 段の決定 2、ア）。


def _window(go):
    async def run():
        sm = PajuState()
        await sm.start()
        await go(sm)

    asyncio.run(asyncio.wait_for(run(), timeout=5.0))


def _almost_lapsed(left: float) -> float:
    """いまから `left` 秒で切れる窓の、開けた時刻。"""
    from familiar_agent.core.wake_window import WAKE_WINDOW_SEC

    return time.monotonic() - WAKE_WINDOW_SEC + left


def test_an_opened_window_is_open_and_lapses_on_time():
    async def go(sm):
        await sm.open_window(_almost_lapsed(0.1))
        assert "open" in _ids(sm)
        assert sm.window_open_at(time.monotonic())
        await asyncio.sleep(0.4)
        assert "shut" in _ids(sm)
        assert not sm.window_open_at(time.monotonic())

    _window(go)


def test_the_window_stays_open_while_speaking_and_lapses_after_the_release():
    """話しているあいだは閉じない。話し終わり（`release_window`）から 10 秒で閉じる（本人「話し終わってから１０秒」）。"""

    async def go(sm):
        await sm.open_window(_almost_lapsed(0.1))
        await sm.hold_window()
        await asyncio.sleep(0.4)
        assert "open" in _ids(sm)  # 話している
        await sm.release_window(_almost_lapsed(0.1))
        assert "open" in _ids(sm)
        await asyncio.sleep(0.4)
        assert "shut" in _ids(sm)

    _window(go)


def test_extending_moves_the_lapse_later():
    async def go(sm):
        await sm.open_window(_almost_lapsed(0.2))
        await sm.extend_window(
            _almost_lapsed(0.8) + 0.0
        )  # 届いた時刻は窓の中（開けた時刻より 0.6 秒あと）
        await asyncio.sleep(0.45)
        assert "open" in _ids(sm)  # 元のままなら、もう閉じている
        await asyncio.sleep(0.7)
        assert "shut" in _ids(sm)

    _window(go)


def test_extending_a_shut_window_does_nothing(caplog):
    """閉じた窓は延ばさない（`WakeWindow.extend` と同じ）。事象も送らない（捨てたログが出ない）。"""

    async def go(sm):
        with caplog.at_level(logging.INFO, logger="familiar_agent.loop.paju_state"):
            await sm.extend_window(time.monotonic())
        assert "shut" in _ids(sm)
        assert not sm.window_open_at(time.monotonic())

    _window(go)
    assert not [r for r in caplog.records if "捨てた" in r.getMessage()]


def test_the_lapse_is_not_sent_when_the_window_is_already_shut(caplog):
    """**番人**：数える仕事は、送る前に状態を見る。閉じているのに `window_lapsed` を送ると、ライブラリが列のほかの
    事象も退けるため（第 1 段 項目 7）。"""

    async def go(sm):
        await sm.open_window(_almost_lapsed(0.1))
        await sm.fire("window_lapsed")  # ほかの道で、先に閉じた
        with caplog.at_level(logging.INFO, logger="familiar_agent.loop.paju_state"):
            await asyncio.sleep(0.4)
        assert "shut" in _ids(sm)

    _window(go)
    assert not [r for r in caplog.records if "捨てた" in r.getMessage()]


def test_a_word_from_inside_the_window_reopens_it_in_the_same_talk():
    """切れる直前に話し始めた声は、`window_lapsed` が先に届いても窓の中。同じ窓の会話のまま開け直す（決定 2）。"""

    async def go(sm):
        await sm.open_window(_almost_lapsed(0.1))
        session = sm.wake.session
        said = time.monotonic()  # 窓の中で話し始めた
        await asyncio.sleep(0.4)
        assert "shut" in _ids(sm)
        assert sm.window_open_at(said)
        await sm.extend_window(said)
        assert "open" in _ids(sm)
        assert sm.wake.session == session
        from familiar_agent.core.wake_window import WAKE_WINDOW_SEC

        assert sm.wake.until == pytest.approx(said + WAKE_WINDOW_SEC)

    _window(go)


def test_ending_the_timers_stops_the_lapse():
    async def go(sm):
        await sm.open_window(_almost_lapsed(0.1))
        sm.end_timers()
        await asyncio.sleep(0.4)
        assert "open" in _ids(sm)

    _window(go)


def test_an_old_arrival_does_not_move_the_window_open():
    """10 秒より前の届いた時刻で開けても、いまはもう切れているので、開いているへ移さない（開いてすぐ閉じない）。"""

    async def go(sm):
        await sm.open_window(time.monotonic() - 20.0)
        assert "shut" in _ids(sm)

    _window(go)


def test_no_timer_starts_after_the_timers_are_ended():
    async def go(sm):
        sm.end_timers()
        await sm.open_window(_almost_lapsed(0.1))
        await asyncio.sleep(0.4)
        assert "open" in _ids(sm)  # 数えていない

    _window(go)


def test_a_window_reopened_while_the_lapse_is_sent_goes_back_open():
    """`window_lapsed` を送っているあいだに開け直されたら、送り終えてからデータを見て、開いているへ戻す。"""

    async def go(sm):
        real_fire = sm.fire

        async def fire(event):
            if event == "window_lapsed":
                sm.wake.open(time.monotonic())  # 送っている最中に、名前で開け直された
            await real_fire(event)

        sm.fire = fire
        await sm.open_window(_almost_lapsed(0.1))
        await asyncio.sleep(0.4)
        assert "open" in _ids(sm)
        assert sm._window_task is not None and not sm._window_task.done()  # 数え直している

    _window(go)


def test_a_failing_chart_does_not_stop_the_window(caplog):
    """状態図が落ちても、会話は止めない。データは開け、警告を 1 行残す。"""

    async def go(sm):
        async def fire(event):
            raise RuntimeError("落ちた")

        sm.fire = fire
        with caplog.at_level(logging.WARNING, logger="familiar_agent.loop.paju_state"):
            await sm.open_window(time.monotonic())
        assert sm.window_open_at(time.monotonic())

    _window(go)
    assert [r for r in caplog.records if "窓を開いているへ移せなかった" in r.getMessage()]
