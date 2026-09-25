"""書き換えた後の道具にも、候補の絞りを当てる（出-ag-ろ 穴 1・2026-09-25）。

道具が返った反復では、返ってきた道具を調停の候補から外す（出-x）。ところが読み取りは、
**候補に照らした後で**道具名を書き換えていた（`set_alarm` に何分後を書いたら `set_timer`、など）。
外したはずの `set_timer` が、`set_alarm` から書き換わって戻ってきた。

実機 2026-09-21 17:34 の反復 2：外したはずの `set_timer` がもう一度投げられた。入力の
`after_minutes=3.0`（小数）は `query` から作ったときの形で、反復 1 は `3`（整数）だった。

候補に無い `set_timer` を直接書いたときは、いまも読めない選択として `full` へ倒れる
（`None`）。書き換えで生まれた場合も同じ扱いにそろえる。
"""

from __future__ import annotations

from familiar_agent.loop.arbiter import _parse

#: 返った `set_timer` を外した候補（ほかのタイマー・アラームの道具は残る）。
WITHOUT_SET_TIMER = (
    "cancel_timer",
    "pause_timer",
    "resume_timer",
    "set_alarm",
    "cancel_alarm",
    "start_stopwatch",
    "stop_stopwatch",
)
WITH_SET_TIMER = ("set_timer", *WITHOUT_SET_TIMER)


def test_an_alarm_rewritten_into_an_excluded_timer_is_refused():
    d = _parse(
        '{"branch":"action","action":"set_alarm","query":"3分"}',
        extra_actions=WITHOUT_SET_TIMER,
    )
    assert d is None, f"外した set_timer が書き換えで戻った：{d}"


def test_the_same_rewrite_still_works_when_the_timer_is_offered():
    """反証：候補に `set_timer` があれば、これまでどおり書き換わる。"""
    d = _parse(
        '{"branch":"action","action":"set_alarm","query":"3分"}',
        extra_actions=WITH_SET_TIMER,
    )
    assert d is not None and d.action == "set_timer"


def test_writing_the_excluded_timer_directly_was_already_refused():
    """そろえる先の振る舞い：直接書いた場合は、いまも倒れる。"""
    d = _parse(
        '{"branch":"action","action":"set_timer","tool_input":{"after_minutes":3,"label":"t"}}',
        extra_actions=WITHOUT_SET_TIMER,
    )
    assert d is None


def test_a_timer_rewritten_into_an_excluded_stop_is_refused():
    """ほかの書き換え（`set_timer` に id だけ → `cancel_timer`）も、行き先が候補に無ければ倒す。"""
    offer = tuple(a for a in WITH_SET_TIMER if a != "cancel_timer")
    d = _parse(
        '{"branch":"action","action":"set_timer","tool_input":{"id":"all"}}',
        extra_actions=offer,
    )
    assert d is None, f"外した cancel_timer が書き換えで戻った：{d}"
