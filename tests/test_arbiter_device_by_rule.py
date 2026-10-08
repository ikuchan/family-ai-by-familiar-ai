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
    "[音楽] 30 分たったから、音楽を止めるね",
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
    jev = jev_says("light")
    asyncio.run(
        decide(
            jev=jev,
            writer=writer_says({"text": "はい"}),
            utterance=_NOTICES[0],
            origin="機器",
            tool_return=True,
        )
    )
    jev.ask.assert_awaited()  # 完了の判定は段 4-2 まで、いまのまま


def test_the_device_lead_no_longer_says_stay_quiet():
    from familiar_agent.loop.arbiter import _LEAD_DEVICE

    assert "黙っていてよい" not in _LEAD_DEVICE
    assert "済んだこと" in _LEAD_DEVICE and "改めて応じない" in _LEAD_DEVICE
