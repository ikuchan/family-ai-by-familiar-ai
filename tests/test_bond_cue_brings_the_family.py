"""BOND の発火で、想起の手がかりに家族の予定とメモの新しい行を入れる（情-q・2026-10-10・本人の決定）。

BOND の発火の W は、内的な促しの文（「誰かと居たい気持ちが湧いている…」）を手がかりに引いていたので、誰が居るか・
これから何があるかと結びつきにくかった。bond のときだけ、手がかりに先読みした 7 日分の予定（先頭 600 字〔仮〕）と、
パジュへのメモで最後に新しく書かれたこと（先頭 300 字）を並べる。O に書く起点の文は変えない。ESTEEM は自尊心なので
人のことに向かない（本人）——いまの手がかりのまま。
"""

from __future__ import annotations

import asyncio

from familiar_agent.core.bond_cue import cue_for
from familiar_agent.loop import notes_watch, schedule_watch

URGE = "[内的な促し:bond] 誰かと居たい気持ちが湧いている。"


def test_the_cue_lines_up_the_schedule_and_the_memo():
    got = cue_for(URGE, "- 10-11（日） サッカー", ["- 金曜はたいきの試合"])
    assert got.startswith(URGE)
    assert "家族のこれから 7 日の予定：\n- 10-11（日） サッカー" in got
    assert "パジュへのメモで新しく書かれたこと：\n- 金曜はたいきの試合" in got


def test_the_parts_are_cut_and_missing_ones_left_out():
    got = cue_for(URGE, "あ" * 1000, ["い" * 500])
    assert "あ" * 600 in got and "あ" * 601 not in got
    assert "い" * 300 in got and "い" * 301 not in got
    assert cue_for(URGE, "", []) == URGE
    assert "メモ" not in cue_for(URGE, "- 予定", [])


def test_the_notes_keep_the_last_added_lines():
    notes_watch._save_state("- 木曜は早く帰る", added=["- 金曜はたいきの試合"])
    assert notes_watch.last_added() == ["- 金曜はたいきの試合"]
    assert notes_watch._load_state() == "- 木曜は早く帰る"


def _begin(axis: str):
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    ip = InformationProcessing(_agent(stream_returns=[]))

    async def no_iterate():
        return ""

    ip._iterate = no_iterate  # type: ignore[method-assign]

    async def go():
        await ip._begin_affect(axis, "気持ちが湧いている。")
        await ip.close()

    asyncio.run(go())
    return ip


def test_a_bond_fire_brings_the_family_into_the_cue():
    schedule_watch._save_state("- 10-11（日） たいき サッカーの試合")
    notes_watch._save_state("- 本文", added=["- 金曜はたいきの試合"])
    ip = _begin("bond")
    assert "サッカーの試合" in ip._req.cue and "金曜はたいきの試合" in ip._req.cue


def test_other_axes_keep_the_urge_as_the_cue():
    schedule_watch._save_state("- 10-11（日） たいき サッカーの試合")
    for axis in ("esteem", "seeking"):
        ip = _begin(axis)
        assert "サッカー" not in ip._req.cue and ip._req.cue.startswith(f"[内的な促し:{axis}]")
