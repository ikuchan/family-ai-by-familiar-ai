"""音楽のおすすめを、話しかけてよいときに 1 日 1 回まで聞き、返事を道具で受ける（知-aa 段 4・2026-10-02・本人の決定）。

- **勧める**：bond・esteem の発火（話しかけてよいとき）で、その日にまだ勧めていなければ、主LLM のシステム文に
  `[音楽のおすすめ]`（曲名・アーティスト・理由）を載せる。「こんな曲あるけどどう？」と聞くかは主LLM が決める。
- **勧めた印**：声にした返事に曲名が入っていたら、機械が「勧めた」と印をつける（勧めた回数・その日・前に勧めた曲）。
- **返事**：勧めたあとの会話では、何を勧めたかを `[音楽のおすすめ]` に載せる。返事は道具 `music_suggestion_reply`
  で受ける——「気に入った」ならかけて気に入った曲に控え、「いらない」ならその曲は二度と勧めない。どちらかは調停が
  意味と動作で決める（出-ay 段 4-4f・`test_music_suggestion_reply_by_jev`）。
- 返事が無いまま 2 回勧めたら、その候補は捨てて、晩に次を用意する。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.core import music_suggestion as ms

TODAY = "2026-10-02"


def _cand(offered: int = 0) -> ms.Candidate:
    """候補は毎回作り直す（勧めた印が回数を書き換えるので、試験どうしで共有しない）。"""
    return ms.Candidate(
        "忘れられないの",
        "サカナクション",
        "spotify:track:wasure",
        "夜のドライブによく入っている人の曲",
        offered,
    )


def test_the_frame_offers_on_a_talking_drive_once_a_day():
    s = ms.Suggestions(candidate=_cand())
    text = ms.frame(s, today=TODAY, talking=True, conversation=False)
    assert text.startswith("[音楽のおすすめ]")
    assert "忘れられないの" in text and "サカナクション" in text and "夜のドライブ" in text
    assert "どう？" in text
    offered_today = ms.Suggestions(candidate=_cand(), last_offered_on=TODAY)
    assert ms.frame(offered_today, today=TODAY, talking=True, conversation=False) == ""


def test_no_frame_for_other_drives_or_without_a_candidate():
    s = ms.Suggestions(candidate=_cand())
    assert ms.frame(s, today=TODAY, talking=False, conversation=False) == ""
    assert ms.frame(ms.Suggestions(), today=TODAY, talking=True, conversation=False) == ""


def test_after_offering_the_conversation_is_told_what_was_offered():
    s = ms.Suggestions(candidate=_cand(1), last_offered_on=TODAY)
    text = ms.frame(s, today=TODAY, talking=False, conversation=True)
    assert "忘れられないの" in text  # 返事の受け方は調停が持つ（出-ay 段 4-4f）
    assert (
        ms.frame(ms.Suggestions(candidate=_cand()), today=TODAY, talking=False, conversation=True)
        == ""
    )


def test_saying_the_title_marks_it_offered():
    s = ms.Suggestions(candidate=_cand())
    assert ms.mark_offered(
        s,
        "パパ、いま少しいい？こんな曲あるけどどう？サカナクションの「忘れられないの」",
        today=TODAY,
    )
    assert s.candidate.offered == 1 and s.last_offered_on == TODAY
    assert "spotify:track:wasure" in s.offered_uris
    other = ms.Suggestions(candidate=_cand())
    assert not ms.mark_offered(other, "おかえりなさい", today=TODAY)


def test_a_candidate_offered_twice_without_a_reply_is_dropped():
    s = ms.Suggestions(candidate=_cand(ms.MAX_OFFERS))
    assert ms.frame(s, today=TODAY, talking=True, conversation=False) == ""
    ms.drop_if_unanswered(s)
    assert s.candidate is None


# ── 返事の道具 ───────────────────────────────────────────────────────────────


@pytest.fixture
def state(monkeypatch):
    data = {"s": ms.Suggestions(candidate=_cand(1), last_offered_on=TODAY)}
    monkeypatch.setattr(ms, "stored", lambda: data["s"])
    monkeypatch.setattr(ms, "store", lambda s: data.__setitem__("s", s) or True)
    return data


def _tool():
    from familiar_agent.tools.music import MusicTool

    io = MagicMock()
    io.play = AsyncMock(return_value=True)
    io.set_shuffle = AsyncMock(return_value=True)
    return MusicTool(io=io, bus=lambda: MagicMock(), table=lambda: ()), io


def test_liking_plays_it_and_keeps_it(state):
    tool, io = _tool()
    text, ok = asyncio.run(tool.call("music_suggestion_reply", {"reply": "気に入った"}))
    assert ok and io.play.await_args.args[1] == "spotify:track:wasure"
    assert "忘れられないの" in text
    assert state["s"].candidate is None
    assert state["s"].liked[-1]["uri"] == "spotify:track:wasure"


def test_declining_never_offers_it_again(state):
    tool, io = _tool()
    text, ok = asyncio.run(tool.call("music_suggestion_reply", {"reply": "いらない"}))
    assert ok and state["s"].candidate is None
    assert state["s"].declined[-1]["uri"] == "spotify:track:wasure"
    io.play.assert_not_awaited()


def test_no_candidate_no_reply(state):
    state["s"] = ms.Suggestions()
    tool, _ = _tool()
    text, ok = asyncio.run(tool.call("music_suggestion_reply", {"reply": "気に入った"}))
    assert not ok and "勧めている曲は無い" in text


def test_the_reply_tool_is_offered_with_the_music_tools():
    from familiar_agent.loop.event_loop import _FULL_ACTIONS

    tool, _ = _tool()
    assert "music_suggestion_reply" in [d["name"] for d in tool.get_tool_definitions()]
    assert "music_suggestion_reply" in _FULL_ACTIONS


# ── ループの結線 ─────────────────────────────────────────────────────────────


def test_a_talking_drive_gets_the_frame_and_saying_the_title_marks_it(state, monkeypatch):
    from familiar_agent.backends import ToolCall
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent, _turn

    state["s"] = ms.Suggestions(candidate=_cand())
    monkeypatch.setattr(ms, "today", lambda: TODAY)
    say = "パパ、いま少しいい？サカナクションの「忘れられないの」って曲、どう？"
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": say})])])

    async def scenario():
        ip = InformationProcessing(a)
        ip._req.trigger_kind = "情動"
        ip._req.fired_axis = "bond"
        ip._req.cue = "誰かと居たい気持ちが湧いている。"
        ip._delivery_block_reason = lambda: ""  # type: ignore[method-assign]
        system = ip._build_system(present_ctx="", workspace_ctx="", iter_ctx="")
        await ip._speak(say)
        await ip.close()
        return system

    system = asyncio.run(scenario())
    assert "[音楽のおすすめ]" in "\n".join(system)
    assert state["s"].candidate.offered == 1 and state["s"].last_offered_on == TODAY
