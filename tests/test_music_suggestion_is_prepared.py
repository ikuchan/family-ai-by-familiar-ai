"""自発のおすすめの候補を、REST の晩に 1 件作る（知-aa 段 4・2026-10-02・本人の決定）。

パジュが自分で音楽を**かける**ことはしない（本人：こわい）。代わりに、晩に 1 件だけ候補を用意し、話しかけてよい
ときに「こんな曲あるけどどう？」と聞く（段 4 の後半）。ここは候補を作る側。

- 材料：プレイリストに入っているアーティスト（多い順）、前に気に入ってもらえた曲、断られた曲。
- フルLLM が、プレイリストに入っていない曲を 1 つ、理由を添えて選ぶ（その人の別の曲でも、似たテイストの人でも）。
  おすすめ・関連アーティストの API は使えない（404・403）ので、似たテイストはフルLLM の知識で選ぶ。
- 機械が検索で実在を確かめ、プレイリストにある曲・前に勧めた曲・断られた曲を除く。確かめられなければ候補無し。
- 候補が残っているあいだは作らない（1 件を使い切ってから次）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.core import music_catalog as mc
from familiar_agent.core import music_suggestion as ms
from familiar_agent.loop.rest_music import prepare_suggestion

CATALOG = mc.Catalog(
    playlists=[
        mc.Playlist(
            "p1",
            "夜のドライブ",
            "spotify:playlist:p1",
            [
                {"title": "Tokyo", "artist": "サカナクション", "uri": "spotify:track:tokyo"},
                {"title": "新宝島", "artist": "サカナクション", "uri": "spotify:track:shin"},
                {"title": "Lemon", "artist": "米津玄師", "uri": "spotify:track:lemon"},
            ],
        )
    ]
)


@pytest.fixture
def state(monkeypatch):
    data: dict = {"s": ms.Suggestions()}
    monkeypatch.setattr(ms, "stored", lambda: data["s"])
    monkeypatch.setattr(ms, "store", lambda s: data.__setitem__("s", s) or True)
    monkeypatch.setattr(mc, "stored", lambda: CATALOG)
    return data


def _agent(reply: str, hits: list):
    a = MagicMock()
    a.backend.complete = AsyncMock(return_value=reply)
    a._music_tool._web.search = MagicMock(return_value=hits)
    return a


REPLY = '{"title": "忘れられないの", "artist": "サカナクション", "reason": "夜のドライブによく入っている人の、まだ入っていない曲"}'
HIT = {"title": "忘れられないの", "artist": "サカナクション", "uri": "spotify:track:wasure"}


def test_a_candidate_is_made_from_my_artists(state):
    a = _agent(REPLY, [HIT])
    assert asyncio.run(prepare_suggestion(a)) is True
    c = state["s"].candidate
    assert (c.title, c.artist, c.uri) == (
        "忘れられないの",
        "サカナクション",
        "spotify:track:wasure",
    )
    assert "まだ入っていない" in c.reason
    prompt = a.backend.complete.await_args.args[0]
    assert prompt.index("サカナクション") < prompt.index("米津玄師")  # 多く入っている人から並べる


def test_the_liked_and_declined_are_told_to_the_llm(state):
    state["s"] = ms.Suggestions(
        liked=[{"title": "好きな曲", "artist": "A", "uri": "u1"}],
        declined=[{"title": "嫌な曲", "artist": "B", "uri": "u2"}],
    )
    a = _agent(REPLY, [HIT])
    asyncio.run(prepare_suggestion(a))
    prompt = a.backend.complete.await_args.args[0]
    assert "好きな曲" in prompt and "嫌な曲" in prompt


@pytest.mark.parametrize(
    "hits",
    [
        [],  # 実在を確かめられない
        [
            {"title": "別の曲", "artist": "サカナクション", "uri": "spotify:track:x"}
        ],  # 名前が合わない
        [
            {"title": "新宝島", "artist": "サカナクション", "uri": "spotify:track:shin"}
        ],  # もうプレイリストにある
    ],
)
def test_an_unconfirmed_or_known_song_is_not_a_candidate(state, hits):
    reply = (
        REPLY
        if hits and hits[0]["title"] != "新宝島"
        else ('{"title": "新宝島", "artist": "サカナクション", "reason": "r"}' if hits else REPLY)
    )
    a = _agent(reply, hits)
    assert asyncio.run(prepare_suggestion(a)) is False
    assert state["s"].candidate is None


def test_a_song_offered_or_declined_before_is_not_offered_again(state):
    state["s"] = ms.Suggestions(offered_uris=["spotify:track:wasure"])
    a = _agent(REPLY, [HIT])
    assert asyncio.run(prepare_suggestion(a)) is False
    state["s"] = ms.Suggestions(
        declined=[
            {"title": "忘れられないの", "artist": "サカナクション", "uri": "spotify:track:wasure"}
        ]
    )
    assert asyncio.run(prepare_suggestion(a)) is False


def test_no_new_candidate_while_one_is_left(state):
    state["s"] = ms.Suggestions(candidate=ms.Candidate("前の曲", "A", "u", "r"))
    a = _agent(REPLY, [HIT])
    assert asyncio.run(prepare_suggestion(a)) is False
    a.backend.complete.assert_not_awaited()


def test_a_bad_reply_makes_no_candidate(state):
    a = _agent("考えつかない", [HIT])
    assert asyncio.run(prepare_suggestion(a)) is False


def test_without_my_artists_nothing_is_prepared(state, monkeypatch):
    monkeypatch.setattr(mc, "stored", lambda: mc.Catalog())
    a = _agent(REPLY, [HIT])
    assert asyncio.run(prepare_suggestion(a)) is False
    a.backend.complete.assert_not_awaited()


def test_the_state_round_trips_in_the_db():
    ms.clear()
    try:
        s = ms.Suggestions(
            candidate=ms.Candidate("曲", "人", "spotify:track:1", "理由", offered=1),
            last_offered_on="2026-10-02",
            liked=[{"title": "a", "artist": "b", "uri": "u"}],
            offered_uris=["spotify:track:0"],
        )
        assert ms.store(s)
        got = ms.stored()
        assert got.candidate == s.candidate and got.last_offered_on == "2026-10-02"
        assert got.liked == s.liked and got.offered_uris == s.offered_uris
    finally:
        ms.clear()
