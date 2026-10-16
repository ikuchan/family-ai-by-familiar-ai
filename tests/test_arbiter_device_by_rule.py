"""機器の知らせは、Jev に聞かず「軽く知らせる」と機械で決める（出-ay 段 4-1・2026-10-09・`設計方針_判定の段` §2.2.5）。

機器の知らせ（タイマーが鳴った・アラームが鳴った・音楽を 30 分で止めた）は 3 つとも「軽く知らせる」と決めた（本人）。
選択肢が 1 つなので Jev には聞かない。軽量LLM が一言を書き、light で返す。書けなければ、いまどおり full（主LLM）。
その求めで道具の結果が届いた反復は「完了」なので、この道を通らない（段 4-2）。
"""

from __future__ import annotations

import asyncio

import pytest

from tests._arbiter_fakes import decide, jev_says, prompt_of, writer_says

_NOTICES = [
    "[タイマー] タイマー：「パパとの約束」の時間（パパに頼まれたもの）",
    "[アラーム] アラーム：7 時（パパに頼まれたもの）",
    "[音楽] 音楽：30 分たったので止めた",
]


@pytest.mark.parametrize("notice", _NOTICES)
def test_a_device_notice_is_told_lightly_without_asking_jev(notice):
    jev = jev_says("full")
    writer = writer_says({"text": "タイマーの時間だよ"})
    d = asyncio.run(decide(jev=jev, writer=writer, utterance=notice, origin="機器"))
    jev.ask.assert_not_awaited()
    assert (d.branch, d.text) == ("light", "タイマーの時間だよ")
    assert writer.complete.await_count == 1
    prompt = prompt_of(writer)
    assert "軽く知らせる" in prompt
    assert '"text"' in prompt and '"query"' not in prompt


def test_if_the_writer_fails_it_falls_back_to_full():
    writer = writer_says(raises=RuntimeError("down"))
    d = asyncio.run(
        decide(jev=jev_says("light"), writer=writer, utterance=_NOTICES[0], origin="機器")
    )
    assert d.branch == "full"


def test_a_tool_return_inside_a_device_request_is_not_this_road():
    """機器の求めの中で道具が返った反復は、機器の道ではなく完了の道を通る（段 4-2・段 5e）。"""
    jev = jev_says("light")
    writer = writer_says({"text": "はい"})
    asyncio.run(
        decide(
            jev=jev,
            writer=writer,
            utterance=_NOTICES[0],
            origin="機器",
            tool_return=True,
            returned=(("cancel_timer", False, "タイマーを止めた"),),
        )
    )
    assert "軽く知らせる" not in prompt_of(writer)  # 機器の道の「決めたこと」ではない
    assert "軽く伝える" in prompt_of(writer)  # 完了の表：止めた → 軽く伝える


def test_the_device_lead_no_longer_says_stay_quiet():
    from familiar_agent.loop.arbiter import _LEAD_DEVICE

    assert "黙っていてよい" not in _LEAD_DEVICE
    assert "済んだこと" in _LEAD_DEVICE and "改めて応じない" in _LEAD_DEVICE


# ── 知らせの理由・時刻を落とさない（知-bh・2026-10-16・本人の決定イ） ──────────

_KEEP = "知らせにある理由・時刻・何のためは落とさずに入れる"


@pytest.mark.parametrize("notice", _NOTICES)
def test_the_writer_is_told_to_keep_the_reason_and_time(notice):
    """実機 10/16 14:03、30 分で止めた知らせが「音楽を止めました。」になり、理由が落ちた。"""
    writer = writer_says({"text": "30 分たったから止めたよ"})
    asyncio.run(decide(jev=jev_says("full"), writer=writer, utterance=notice, origin="機器"))
    assert _KEEP in prompt_of(writer)


def test_the_other_roads_are_not_told_the_device_rule():
    """**反証の側**：機器の知らせ以外（返事・情動・道具の返り）の先導文には載せない。"""
    from familiar_agent.loop import arbiter

    for lead in (arbiter._LEAD_REPLY, arbiter._LEAD_SELF, arbiter._LEAD_TOOL_RETURN):
        assert _KEEP not in lead
    writer = writer_says({"text": "はい"})
    asyncio.run(
        decide(
            jev=jev_says("light"),
            writer=writer,
            utterance=_NOTICES[0],
            origin="機器",
            tool_return=True,
            returned=(("cancel_timer", False, "タイマーを止めた"),),
        )
    )
    assert _KEEP not in prompt_of(writer)


def test_jev_hears_the_notices_that_actually_reach_it():
    """Jev に渡す機器の文は、調停に届く知らせだけを並べる（知-bh・2026-10-17・本人の決定ア）。

    人の出入りとメモは記録するだけで調停に届かない（`設計方針_判定の段` §2.2.5）。書き手の先導文と並びをそろえる。
    """
    from familiar_agent.loop.arbiter import _JEV_LEAD, _LEAD_DEVICE

    lead = _JEV_LEAD["機器"]
    assert "（タイマー・アラーム・音楽）" in lead and "（タイマー・アラーム・音楽）" in _LEAD_DEVICE
    assert "人の出入り" not in lead and "メモ" not in lead
