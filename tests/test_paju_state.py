"""環-aw 第 1 段 項目 1：パジュの状態図（頭と耳）の骨組みと、捨てた事象のログ（`設計方針_パジュの状態` v0.6 決定ス、ユ）。

頭は何もしていない、考えている（子 5 つ）、答えを言っている。耳は聞いている、考えているあいだ聞かない。耳は頭に
入るときの動作ではなく事象で動く（答えのあと考えているへ戻っても、聞いたまま）。
"""

from __future__ import annotations

import asyncio
import contextlib
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
        real_fire = sm._fire

        async def fire(event):
            if event == "window_lapsed":
                sm.wake.open(time.monotonic())  # 送っている最中に、名前で開け直された
            await real_fire(event)

        sm._fire = fire
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

        sm._fire = fire
        with caplog.at_level(logging.WARNING, logger="familiar_agent.loop.paju_state"):
            await sm.open_window(time.monotonic())
        assert sm.window_open_at(time.monotonic())

    _window(go)
    assert [r for r in caplog.records if "窓を開いているへ移せなかった" in r.getMessage()]


# ---- 環-aw 第 2 段 項目 3：仮の呼びかけ ----
# 名前だけで確かさの足りなかった声（知-bs）。届いた時刻から 10 秒で閉じているへ戻る。続きかは届いた時刻のデータで決める。


def test_a_tentative_call_lapses_ten_seconds_after_it_arrived():
    """数え始めは届いた時刻で、処理した時刻ではない。切れたあとも、届いた時刻は残す（続きかを決めるのに使う）。"""

    async def go(sm):
        at = _almost_lapsed(0.1)
        await sm.call_tentatively(at)
        assert "tentative" in _ids(sm) and sm.tentative_since() == at
        await asyncio.sleep(0.4)
        assert "shut" in _ids(sm)
        assert sm.tentative_since() == at

    _window(go)


def test_an_old_tentative_call_does_not_move_the_state():
    """**反証の側**：10 秒より前に届いた声では、仮の呼びかけへ移らない（移ってすぐ戻らない）。"""

    async def go(sm):
        await sm.call_tentatively(time.monotonic() - 20.0)
        assert "shut" in _ids(sm)

    _window(go)


def test_a_window_opened_from_a_tentative_call_keeps_its_arrival(caplog):
    """窓が開くと、仮の呼びかけから 1 回で開いているへ移る。届いた時刻は捨てない（2026-11-12 本人の決定）。"""

    async def go(sm):
        at = time.monotonic()
        await sm.call_tentatively(at)
        with caplog.at_level(logging.DEBUG, logger="familiar_agent.loop.paju_state"):
            await sm.open_window(time.monotonic())
        assert "open" in _ids(sm)
        assert sm.tentative_since() == at

    _window(go)
    moved = [r.getMessage() for r in caplog.records if "で移った" in r.getMessage()]
    assert len(moved) == 1 and "window_opened" in moved[0]


def test_dropping_the_tentative_call_shuts_it():
    async def go(sm):
        await sm.call_tentatively(time.monotonic())
        await sm.drop_tentative()
        assert "shut" in _ids(sm) and sm.tentative_since() is None

    _window(go)


def test_dropping_without_a_tentative_call_sends_nothing(caplog):
    """**番人**：仮の呼びかけにいないときは、`tentative_dropped` を送らない（移りの無い事象は列のほかの事象も退ける）。"""

    async def go(sm):
        with caplog.at_level(logging.INFO, logger="familiar_agent.loop.paju_state"):
            await sm.drop_tentative()
        assert "shut" in _ids(sm)

    _window(go)
    assert not [r for r in caplog.records if "捨てた" in r.getMessage()]


def test_ending_the_timers_stops_the_tentative_lapse():
    async def go(sm):
        await sm.call_tentatively(_almost_lapsed(0.1))
        sm.end_timers()
        await asyncio.sleep(0.4)
        assert "tentative" in _ids(sm)

    _window(go)


def test_a_tentative_call_is_kept_while_the_state_is_open(caplog):
    """門は届いた時刻の窓で決めるので、状態が開いていても届いた時刻は預ける（前と同じ）。事象は送らない。"""

    async def go(sm):
        await sm.open_window(time.monotonic())
        at = time.monotonic()
        with caplog.at_level(logging.INFO, logger="familiar_agent.loop.paju_state"):
            await sm.call_tentatively(at)
        assert "open" in _ids(sm) and sm.tentative_since() == at

    _window(go)
    assert not [r for r in caplog.records if "捨てた" in r.getMessage()]


def test_a_lapse_being_sent_is_finished_even_if_stopped():
    """切れたことを送っている印のあいだは、数える仕事を止めない（送り終える）。"""

    async def go(sm):
        real_fire = sm._fire

        async def fire(event):
            if event == "tentative_lapsed":
                # 送っている最中に、別の道（仕事の外）が止めた
                asyncio.get_running_loop().call_soon(sm._stop_tentative_task)
                await asyncio.sleep(0)
                await asyncio.sleep(0)
            await real_fire(event)

        sm._fire = fire
        await sm.call_tentatively(_almost_lapsed(0.1))
        await asyncio.sleep(0.4)
        assert "shut" in _ids(sm)

    _window(go)


def test_dropping_while_the_lapse_is_sent_sends_nothing(caplog):
    """**番人**：切れたことを送っている最中に捨てても、`tentative_dropped` は送らない（送り終えれば閉じているにいる）。"""

    async def go(sm):
        real_fire = sm._fire
        late: list = []

        async def fire(event):
            if event == "tentative_lapsed":
                # 送っている最中に、音楽の言葉を受けた。順番を待つので、送り終えてから処理される（決定ン）
                late.append(asyncio.ensure_future(sm.drop_tentative()))
            await real_fire(event)

        sm._fire = fire
        await sm.call_tentatively(_almost_lapsed(0.1))
        with caplog.at_level(logging.INFO, logger="familiar_agent.loop.paju_state"):
            await asyncio.sleep(0.4)
            await asyncio.gather(*late)
        assert late and "shut" in _ids(sm) and sm.tentative_since() is None

    _window(go)
    assert not [r for r in caplog.records if "捨て" in r.getMessage()]


def test_a_call_while_the_lapse_is_sent_goes_back_to_tentative():
    """切れたことを送っている最中に新しい呼びかけが届いたら、送り終えてからデータを見て、仮の呼びかけへ戻す。"""

    async def go(sm):
        real_fire = sm._fire
        late: list = []

        async def fire(event):
            if event == "tentative_lapsed":
                # 送っている最中に届いた。順番を待つので、送り終えてから処理される（決定ン）
                late.append(asyncio.ensure_future(sm.call_tentatively(time.monotonic())))
            await real_fire(event)

        sm._fire = fire
        await sm.call_tentatively(_almost_lapsed(0.1))
        await asyncio.sleep(0.4)
        await asyncio.gather(*late)
        assert late and "tentative" in _ids(sm)
        assert sm._tentative_task is not None and not sm._tentative_task.done()

    _window(go)


def test_a_call_kept_in_the_window_moves_to_tentative_when_the_window_lapses():
    """窓が切れる間際に届いた仮の呼びかけは、窓が切れたら仮の呼びかけへ移る（状態とデータをそろえる）。"""

    async def go(sm):
        await sm.open_window(_almost_lapsed(0.1))
        await sm.call_tentatively(time.monotonic())
        assert "open" in _ids(sm)
        await asyncio.sleep(0.4)
        assert "tentative" in _ids(sm)

    _window(go)


def test_an_old_tentative_call_starts_no_timer():
    async def go(sm):
        await sm.call_tentatively(time.monotonic() - 20.0)
        assert sm._tentative_task is None

    _window(go)


def test_a_window_opened_from_a_tentative_call_stops_its_timer():
    async def go(sm):
        await sm.call_tentatively(time.monotonic())
        await sm.open_window(time.monotonic())
        assert sm._tentative_task is None

    _window(go)


# ---- 環-aw 第 2 段 項目 4：黙っている（口の黙るか） ----
# 持ち主は状態図。DB の `agent_state` の行は控えで、入ると書き、出ると消し、起動のときに残りの期限で戻す。


def _hushing(go, *, restore=None):
    """黙るかを確かめる装置。控えを書く口と消す口の呼ばれ方を `book` に残す。"""
    book: dict = {"saved": [], "cleared": 0, "loaded": 0}

    def saved(req):
        book["saved"].append(req)

    def cleared():
        book["cleared"] += 1

    def load():
        book["loaded"] += 1
        return restore

    async def run():
        sm = PajuState(on_silence_saved=saved, on_silence_cleared=cleared, restore_silence=load)
        await sm.start()
        await go(sm, book)

    asyncio.run(asyncio.wait_for(run(), timeout=5.0))
    return book


def _req(left: float, reason: str = ""):
    from familiar_agent.silence_state import SilenceRequest

    return SilenceRequest(person="パパ", until=time.time() + left, reason=reason)


def test_hushing_makes_the_mouth_silent_and_saves_the_request():
    req = _req(600)

    async def go(sm, book):
        await sm.hush_with(req)
        assert "silent" in _ids(sm)
        assert sm.is_silent() and sm.silence_request() == req
        assert book["saved"] == [req]

    _hushing(go)


def test_the_silence_ends_on_time_and_clears_the_request():
    async def go(sm, book):
        await sm.hush_with(_req(0.1))
        await asyncio.sleep(0.4)
        assert "may_speak" in _ids(sm)
        assert sm.silence_request() is None and book["cleared"] == 1

    _hushing(go)


def test_unhushing_frees_the_mouth_and_clears_the_request():
    async def go(sm, book):
        await sm.hush_with(_req(600))
        await sm.unhush()
        assert "may_speak" in _ids(sm) and not sm.is_silent()
        assert sm.silence_request() is None and book["cleared"] == 1

    _hushing(go)


def test_unhushing_while_free_sends_nothing(caplog):
    """**番人**：話してよいにいるときは `silence_ended` を送らない（移りの無い事象は列のほかの事象も退ける）。"""

    async def go(sm, book):
        with caplog.at_level(logging.INFO, logger="familiar_agent.loop.paju_state"):
            await sm.unhush()
        assert "may_speak" in _ids(sm)
        assert book["cleared"] == 0

    _hushing(go)
    assert not [r for r in caplog.records if "捨てた" in r.getMessage()]


def test_a_new_request_while_silent_replaces_it_and_its_end():
    """黙っているあいだの新しい依頼（遅いタイマー）は、データも期限も置き換える。"""
    later = _req(0.6, reason="timer:2")

    async def go(sm, book):
        await sm.hush_with(_req(0.1, reason="timer:1"))
        await sm.hush_with(later)
        await asyncio.sleep(0.3)
        assert "silent" in _ids(sm) and sm.silence_request() == later  # 元の期限は過ぎた
        await asyncio.sleep(0.6)
        assert "may_speak" in _ids(sm)

    book = _hushing(go)
    assert len(book["saved"]) == 2


def test_a_live_request_is_restored_at_start_with_its_end():
    """再起動しなおし：DB に生きた依頼があれば、`start` のあと、残りの期限で黙っている。控えは書き直さない。"""
    req = _req(0.3)

    async def go(sm, book):
        assert "silent" in _ids(sm) and sm.silence_request() == req
        await asyncio.sleep(0.6)
        assert "may_speak" in _ids(sm)

    book = _hushing(go, restore=req)
    assert book["saved"] == [] and book["loaded"] == 1


def test_an_expired_request_is_cleared_at_start():
    """**反証の側**：期限を過ぎた控えは消し、話してよいにいる。"""

    async def go(sm, book):
        assert "may_speak" in _ids(sm) and sm.silence_request() is None
        assert book["cleared"] == 1

    _hushing(go, restore=_req(-1))


def test_start_restores_once_even_when_awaited_together():
    """戻し終えるまで、ほかの読み手も待つ（門とトニックが同時に `start` を待っても、読むのは 1 度）。"""
    req = _req(600)
    loaded = []

    def load():
        loaded.append(1)
        return req

    async def run():
        sm = PajuState(restore_silence=load)
        await asyncio.gather(sm.start(), sm.start())
        assert sm.is_silent()

    asyncio.run(asyncio.wait_for(run(), timeout=5.0))
    assert loaded == [1]


def test_the_window_can_open_while_silent():
    """例 1、例 2：黙っているあいだも耳は聞いていて、窓は開けられる（窓と黙るかは別の領域）。"""

    async def go(sm, book):
        await sm.hush_with(_req(600))
        await sm.open_window(time.monotonic())
        assert {"open", "silent"} <= _ids(sm)

    _hushing(go)


def test_ending_the_timers_stops_the_silence_end():
    async def go(sm, book):
        await sm.hush_with(_req(0.1))
        sm.end_timers()
        await asyncio.sleep(0.4)
        assert "silent" in _ids(sm)

    _hushing(go)


def test_a_chart_entered_by_another_event_still_restores_at_start():
    """ほかの口が先に事象を送り、ライブラリがひとりでに初めの状態に入っても、`start` で控えを戻す。"""
    req = _req(600)

    async def run():
        sm = PajuState(restore_silence=lambda: req)
        await sm.fire("window_opened")
        await sm.start()
        assert sm.is_silent() and sm.silence_request() == req

    asyncio.run(asyncio.wait_for(run(), timeout=5.0))


def test_a_failed_restore_is_tried_again_at_the_next_start():
    """戻す途中で落ちたら、済んだことにしない。次の `start` でやり直す。"""
    req = _req(600)

    async def run():
        sm = PajuState(restore_silence=lambda: req)
        real = sm._restore
        failed: list[int] = []

        async def restore():
            if not failed:
                failed.append(1)
                raise RuntimeError("落ちた")
            await real()

        sm._restore = restore  # type: ignore[method-assign]
        with pytest.raises(RuntimeError):
            await sm.start()
        await sm.start()
        assert sm.is_silent()

    asyncio.run(asyncio.wait_for(run(), timeout=5.0))


def test_a_reader_waiting_on_a_failed_restore_restores_again_itself():
    """同時に待っていた読み手は、戻しが落ちても「戻し終えた」とみなさず、自分で戻し直してから読む。"""
    req = _req(600)

    async def run():
        sm = PajuState(restore_silence=lambda: req)
        real = sm._restore
        failed: list[int] = []

        async def restore():
            if not failed:
                failed.append(1)
                await asyncio.sleep(0.01)  # 二人目が待ち始めるまで
                raise RuntimeError("落ちた")
            await real()

        sm._restore = restore  # type: ignore[method-assign]
        first, second = await asyncio.gather(sm.start(), sm.start(), return_exceptions=True)
        assert isinstance(first, RuntimeError) and second is None
        assert sm.is_silent()  # 二人目が読む前に戻っている

    asyncio.run(asyncio.wait_for(run(), timeout=5.0))


def test_unhushing_frees_a_mouth_left_silent_by_a_failed_end():
    """期限で解けたことを送れず、データが無いのに黙っているに残っても、`unhush` で話してよいへ戻す。"""

    async def go(sm, book):
        real_fire = sm._fire
        failed: list[int] = []

        async def fire(event):
            if event == "silence_ended" and not failed:
                failed.append(1)
                raise RuntimeError("落ちた")
            await real_fire(event)

        sm._fire = fire
        await sm.hush_with(_req(0.1))
        await asyncio.sleep(0.4)
        assert "silent" in _ids(sm) and sm.silence_request() is None
        await sm.unhush()
        assert "may_speak" in _ids(sm)

    _hushing(go)


# ── 送る口は 1 つの順番待ちに並ぶ（環-aw 第 2 段のあと・ウの 1・決定ン）──────────────────────────
# ライブラリは事象の処理の途中で動作を `asyncio.gather` で呼び、ほかの仕事に順番を譲る。そのあいだに並んだ事象は
# まだ処理されていないので、ほかの口から見える状態は古い。古い状態を見て送った事象は移りが無く、ライブラリは列に
# 並んでいたほかの事象も一緒に退ける。


def test_a_lapse_behind_an_opened_window_does_not_drop_the_next_head_event():
    """仮の呼びかけが切れて送るのと窓が開くのが、頭の事象の処理中に重なっても、後ろの頭の事象は捨てない。"""
    from familiar_agent.core.wake_window import WAKE_WINDOW_SEC

    async def go(sm):
        await sm.call_tentatively(time.monotonic())
        sm._stop_tentative_task()
        sm.tentative_at = (
            time.monotonic() - WAKE_WINDOW_SEC
        )  # もう切れている。数える仕事は、すぐ見て送る
        tasks = [
            asyncio.ensure_future(sm.fire("request_opened")),  # 処理の途中で順番を譲る
            asyncio.ensure_future(sm.open_window(time.monotonic())),  # 仮の呼びかけから開いているへ
            asyncio.ensure_future(sm._wait_tentative()),  # 本物の数える仕事
            asyncio.ensure_future(sm.fire("decide_began")),  # 後ろに並ぶ頭の事象
        ]
        await asyncio.gather(*tasks)
        assert {"open", "deciding"} <= _ids(sm)

    _window(go)


def test_hush_and_unhush_together_leave_no_backup_behind():
    """「黙って」と「話していいよ」が頭の事象の処理中に重なっても、話してよいにいて、控えも残らない。"""
    from familiar_agent.silence_state import SilenceRequest

    row: dict = {}  # DB の控えの行の代わり

    def saved(req: SilenceRequest) -> None:
        row["req"] = req

    def cleared() -> None:
        row.pop("req", None)

    def chart() -> PajuState:
        return PajuState(
            on_silence_saved=saved,
            on_silence_cleared=cleared,
            restore_silence=lambda: row.get("req"),
        )

    async def go():
        sm = chart()
        await sm.start()
        await asyncio.gather(
            sm.fire("request_opened"),  # 処理の途中で順番を譲る
            sm.hush_with(_req(600)),
            sm.unhush(),
        )
        assert "may_speak" in _ids(sm) and sm.silence_request() is None
        again = chart()  # 起動し直したときの状態図
        await again.start()
        assert "may_speak" in _ids(again)
        for m in (sm, again):
            m.end_timers()
            m.end_counting()

    asyncio.run(asyncio.wait_for(go(), timeout=5.0))


def test_a_timer_queued_behind_a_person_does_not_override_the_person():
    """人の「黙って」が順番を待つあいだにタイマーを掛けても、タイマーの依頼は人の依頼を上書きしない。依頼は、呼び手が
    読んで決めたその場で置く（決定ン）。"""
    from familiar_agent.silence_state import hush_for_timer

    async def go():
        sm = PajuState()
        await sm.start()
        person = _req(3600)

        async def timer() -> None:  # タイマーを掛ける口と同じ形（読んで決めて置く）
            current = sm.silence_request()
            nxt = hush_for_timer(current, person="ママ", until=time.time() + 60, tid=1)
            if nxt is not current:
                await sm.hush_with(nxt)

        await asyncio.gather(
            sm.fire("request_opened"),  # 処理の途中で順番を譲る
            sm.hush_with(person),
            timer(),
        )
        assert sm.silence_request() is person
        assert "silent" in _ids(sm)
        sm.end_timers()
        sm.end_counting()

    asyncio.run(asyncio.wait_for(go(), timeout=5.0))


def test_the_turn_survives_a_new_event_loop():
    """**番人**：状態図を 1 つ目のループで始め、2 つ目のループで送る口を取り合わせても落ちない（試験の装置と画面は、
    状態図をループより長く使う）。"""
    sm = PajuState()

    async def contend():
        await sm.start()
        await asyncio.gather(
            sm.fire("request_opened"), sm.fire("decide_began"), sm.fire("request_closed")
        )

    asyncio.run(asyncio.wait_for(contend(), timeout=5.0))
    asyncio.run(asyncio.wait_for(contend(), timeout=5.0))
    assert "idle" in _ids(sm)


def test_a_sender_cancelled_mid_send_still_finishes_the_transition():
    """送っている最中に呼び手が取り消されても、その事象の移りは最後まで済み、次の送り手も送れる（決定ン）。"""

    async def go():
        sm = PajuState()
        await sm.start()
        sender = asyncio.ensure_future(sm.fire("request_opened"))
        for _ in range(3):
            await asyncio.sleep(0)  # 処理の途中（動作の gather）まで進める
        sender.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sender
        await sm.fire("decide_began")
        assert {"deciding", "deaf"} <= _ids(sm)
        sm.end_timers()
        sm.end_counting()

    asyncio.run(asyncio.wait_for(go(), timeout=5.0))


def test_a_call_after_a_queued_drop_is_kept():
    """**番人**：捨てる口が順番を待つあいだに届いた呼びかけは、捨てない（届いた時刻はすぐ書き、すぐ消す）。"""

    async def go(sm):
        await sm.call_tentatively(time.monotonic())
        tasks = [
            asyncio.ensure_future(sm.fire("request_opened")),  # 順番を持って処理の途中で譲る
            asyncio.ensure_future(sm.drop_tentative()),  # 音楽の言葉を受けた（順番を待つ）
            asyncio.ensure_future(
                sm.call_tentatively(time.monotonic())
            ),  # そのあとに届いた呼びかけ
        ]
        await asyncio.gather(*tasks)
        assert sm.tentative_since() is not None and "tentative" in _ids(sm)

    _window(go)
