"""すすめた曲への返事は、発話の意味と動作の問いで Jev が受ける（出-ay 段 4-4f・2026-10-10・`設計方針_判定の段` §2.2.5）。

返事のあとは、かけるか、かけないかだけなので Jev に決めさせる（本人）。返事を待っているとき（候補があり、今日すすめた）だけ、
意味に「すすめた曲への返事」を確認待ちへの答えの次に並べ、2 回目は かける／かけない／聞き返す／黙る。決まったら
`music_suggestion_reply` に返事を入れて投げる（書く言葉が無いので軽量LLM は呼ばない）。かけたあとは黙り、「いらない」の
あとは軽く伝える（本人の決定ア）。主LLM の枠からは返事の受け方の行を外す。かけ方は play_music と同じく Web API で確かめる。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.core import music_suggestion as ms
from familiar_agent.core import utterance_meaning as um
from familiar_agent.core.completion_kind import kind_of
from familiar_agent.tools import music as music_mod
from tests._arbiter_fakes import decide, writer_says

TODAY = "2026-10-10"


def _cand(offered: int = 1) -> ms.Candidate:
    return ms.Candidate(
        "忘れられないの", "サカナクション", "spotify:track:wasure", "夜の曲", offered
    )


# ── 返事を待っているか・枠 ──────────────────────────────────────────────────


def test_a_reply_is_awaited_only_on_the_day_it_was_offered():
    assert ms.awaiting_reply(ms.Suggestions(candidate=_cand(), last_offered_on=TODAY), today=TODAY)
    assert not ms.awaiting_reply(
        ms.Suggestions(candidate=_cand(), last_offered_on="2026-10-09"), today=TODAY
    )
    assert not ms.awaiting_reply(ms.Suggestions(candidate=_cand(0)), today=TODAY)
    assert not ms.awaiting_reply(ms.Suggestions(), today=TODAY)


def test_the_main_llm_is_no_longer_told_to_take_the_reply():
    s = ms.Suggestions(candidate=_cand(), last_offered_on=TODAY)
    text = ms.frame(s, today=TODAY, talking=False, conversation=True)
    assert "忘れられないの" in text and "music_suggestion_reply" not in text


# ── 意味と動作 ───────────────────────────────────────────────────────────────


def test_the_reply_meaning_comes_after_confirm_and_only_while_awaited():
    got = um.offered(confirming=True, suggesting=True, music=True, camera=True)
    assert got[:2] == ["confirm", "suggestion"]
    assert "suggestion" not in um.offered(confirming=True, music=True, camera=True)
    assert set(um.ACTIONS_BY_MEANING["suggestion"]) == {
        "suggestion_like",
        "suggestion_decline",
        "ask_back",
        "silent",
    }


def _jev(action: str) -> MagicMock:
    jev = MagicMock()
    jev.available = True
    jev.ask = AsyncMock(
        return_value=JevAnswer(
            ok=True,
            answers={
                "meaning": {"choice": "suggestion", "confidence": 0.9},
                "action_suggestion": {"choice": action, "confidence": 0.9},
            },
        )
    )
    return jev


def _decide(action: str, extra=("play_music", "music_suggestion_reply")):
    writer = writer_says({"text": "x"})
    jev = _jev(action)
    d = asyncio.run(
        decide(
            jev=jev, writer=writer, utterance="いいね、かけて", origin="発話", extra_actions=extra
        )
    )
    return d, jev, writer


def test_liking_sends_the_reply_tool_without_writing():
    d, jev, writer = _decide("suggestion_like")
    assert "action_suggestion" in jev.ask.await_args.args[1]
    assert (d.branch, d.action, d.tool_input) == (
        "action",
        "music_suggestion_reply",
        {"reply": "気に入った"},
    )
    writer.complete.assert_not_awaited()


def test_declining_sends_the_reply_tool_with_no():
    d, _, _ = _decide("suggestion_decline")
    assert (d.action, d.tool_input) == ("music_suggestion_reply", {"reply": "いらない"})


def test_without_a_pending_suggestion_the_meaning_is_not_offered():
    _, jev, _ = _decide("suggestion_like", extra=("play_music",))
    assert "action_suggestion" not in jev.ask.await_args.args[1]


# ── 調停の候補（ループ）──────────────────────────────────────────────────────


@pytest.fixture
def state(monkeypatch):
    data = {"s": ms.Suggestions()}
    monkeypatch.setattr(ms, "stored", lambda: data["s"])
    monkeypatch.setattr(ms, "store", lambda s: data.__setitem__("s", s) or True)
    monkeypatch.setattr(ms, "today", lambda: TODAY)
    return data


def _extra_actions():
    from familiar_agent.loop.event_loop import InformationProcessing
    from familiar_agent.tools.music import MusicTool
    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    a._music_tool = MusicTool(io=MagicMock(), bus=lambda: MagicMock(), table=lambda: ())
    ip = InformationProcessing(a)
    got = ip._extra_actions()
    asyncio.run(ip.close())
    return got


def test_the_arbiter_gets_the_reply_tool_only_while_a_reply_is_awaited(state):
    assert "music_suggestion_reply" not in _extra_actions()
    state["s"] = ms.Suggestions(candidate=_cand(), last_offered_on=TODAY)
    assert "music_suggestion_reply" in _extra_actions()


# ── 完了 ─────────────────────────────────────────────────────────────────────


def test_after_playing_it_is_silent_and_after_no_it_tells_lightly():
    played = kind_of(
        "music_suggestion_reply",
        failed=False,
        result="すすめた「忘れられないの」（サカナクション）をかけ始めた",
        origin="発話",
    )
    declined = kind_of(
        "music_suggestion_reply",
        failed=False,
        result="「忘れられないの」はもう勧めない",
        origin="発話",
    )
    assert played is not None and played[1] == ("silent",)
    assert declined is not None and declined[1] == ("tell_light",)


# ── かけ方 ───────────────────────────────────────────────────────────────────


@pytest.fixture
def _quick(monkeypatch):
    monkeypatch.setattr(music_mod, "_CONFIRM_SEC", 0.2)
    monkeypatch.setattr(music_mod, "_CONFIRM_TICK", 0.05)


def test_liking_plays_through_the_web_and_is_confirmed(state, _quick):
    state["s"] = ms.Suggestions(candidate=_cand(), last_offered_on=TODAY)
    io = MagicMock()
    io.play = AsyncMock(return_value=True)
    web = MagicMock()
    web.play = MagicMock(return_value=True)
    web.now_playing = MagicMock(
        return_value={"playing": True, "item": "spotify:track:wasure", "context": None}
    )
    tool = music_mod.MusicTool(
        io=io, bus=lambda: MagicMock(), table=lambda: (), web=web, device_name="パジュ"
    )
    text, ok = asyncio.run(tool.call("music_suggestion_reply", {"reply": "気に入った"}))
    assert ok and "忘れられないの" in text
    web.play.assert_called_once_with("パジュ", "spotify:track:wasure")
    io.play.assert_not_awaited()
