"""「〇〇かけて」を、近いところから順に探してかける（知-aa 段 3・2026-10-02・本人の決定）。

正解は、あなたが作ったプレイリストや、そこに入っているアーティストであることが多い（本人：ケイマンの正解は自分の
プレイリスト）。Spotify 全体の検索をいきなり使わず、近いところから順に探す。

1. `MUSIC.md`（表記ゆれを均して照らす：けいまん → ケイマン）
2. 自分のプレイリスト（名前で・晩に読んだ目録から）
3. 自分のライブラリ（保存したアルバムと曲）
4. プレイリストに入っているアーティストと、その人の曲
5. Spotify 全体の検索（主LLM が指した種類で）

返りで、何をどこから見つけたかを言う（違えば言い直してもらえる）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.core import music_catalog as mc
from familiar_agent.tools.music import MusicTool

TABLE = (("ケイマン", "spotify:playlist:aaa", False),)

CATALOG = mc.Catalog(
    playlists=[
        mc.Playlist(
            "p1",
            "夜のドライブ",
            "spotify:playlist:p1",
            [
                {"title": "Tokyo", "artist": "サカナクション", "uri": "spotify:track:tokyo"},
                {"title": "新宝島", "artist": "サカナクション", "uri": "spotify:track:shin"},
            ],
        )
    ],
    albums=[{"title": "STRAY SHEEP", "artist": "米津玄師", "uri": "spotify:album:stray"}],
    tracks=[{"title": "Pale Blue", "artist": "米津玄師", "uri": "spotify:track:pale"}],
)


def _tool(monkeypatch, *, search=None):
    monkeypatch.setattr(mc, "stored", lambda: CATALOG)
    io = MagicMock()
    io.play = AsyncMock(return_value=True)
    io.set_shuffle = AsyncMock(return_value=True)
    web = MagicMock()
    web.search = MagicMock(side_effect=search or (lambda q, kind, **kw: []))
    tool = MusicTool(io=io, bus=lambda: MagicMock(), table=lambda: TABLE, web=web)
    return tool, io, web


def _play(tool, **inp):
    return asyncio.run(tool.call("play_music", inp))


def test_music_md_is_found_through_a_loose_reading(monkeypatch):
    tool, io, web = _tool(monkeypatch)
    text, ok = _play(tool, name="けいまん")
    assert ok and io.play.await_args.args[1] == "spotify:playlist:aaa"
    assert "ケイマン" in text
    web.search.assert_not_called()


def test_my_playlist_by_name(monkeypatch):
    tool, io, _ = _tool(monkeypatch)
    text, ok = _play(tool, name="夜のドライブ")
    assert ok and io.play.await_args.args[1] == "spotify:playlist:p1"
    assert "プレイリスト" in text


def test_my_library_comes_before_the_whole_of_spotify(monkeypatch):
    tool, io, web = _tool(monkeypatch)
    _, ok = _play(tool, name="STRAY SHEEP")
    assert ok and io.play.await_args.args[1] == "spotify:album:stray"
    _, ok = _play(tool, name="Pale Blue")
    assert io.play.await_args.args[1] == "spotify:track:pale"
    web.search.assert_not_called()


def test_a_song_in_my_playlists_is_played_as_is(monkeypatch):
    tool, io, web = _tool(monkeypatch)
    text, ok = _play(tool, name="新宝島")
    assert ok and io.play.await_args.args[1] == "spotify:track:shin"
    assert "サカナクション" in text and "プレイリスト" in text
    web.search.assert_not_called()


def test_an_artist_in_my_playlists_is_searched_and_matched_by_name(monkeypatch):
    def search(q, kind, **kw):
        assert kind == "artist"
        return [
            {"title": "サカナクション トリビュート", "artist": "", "uri": "spotify:artist:fake"},
            {"title": "サカナクション", "artist": "", "uri": "spotify:artist:sakana"},
        ]

    tool, io, _ = _tool(monkeypatch, search=search)
    text, ok = _play(tool, name="サカナクション")
    assert ok and io.play.await_args.args[1] == "spotify:artist:sakana"
    assert "プレイリストに入っている" in text


def test_the_whole_of_spotify_prefers_my_artists(monkeypatch):
    def search(q, kind, **kw):
        assert (q, kind) == ("Lemon", "track")
        return [
            {"title": "Lemon", "artist": "誰か", "uri": "spotify:track:other"},
            {"title": "Lemon", "artist": "米津玄師", "uri": "spotify:track:lemon"},
        ]

    tool, io, _ = _tool(monkeypatch, search=search)
    text, ok = _play(tool, name="Lemon", kind="曲")
    assert (
        ok and io.play.await_args.args[1] == "spotify:track:lemon"
    )  # ライブラリに居る米津玄師を先に
    assert "Spotify から探した" in text


def test_nothing_found_is_refused(monkeypatch):
    tool, io, _ = _tool(monkeypatch)
    text, ok = _play(tool, name="存在しない曲", kind="曲")
    assert not ok and "見つからなかった" in text
    io.play.assert_not_awaited()


def test_the_tool_says_what_kind_can_be_named(monkeypatch):
    tool, _, _ = _tool(monkeypatch)
    d = next(d for d in tool.get_tool_definitions() if d["name"] == "play_music")
    assert "kind" in d["input_schema"]["properties"]


@pytest.mark.parametrize(
    "kind,want",
    [
        ("曲", "track"),
        ("アーティスト", "artist"),
        ("アルバム", "album"),
        ("プレイリスト", "playlist"),
        ("", "track"),
    ],
)
def test_the_kind_is_read_in_words(monkeypatch, kind, want):
    seen = {}

    def search(q, k, **kw):
        seen["k"] = k
        return []

    tool, _, _ = _tool(monkeypatch, search=search)
    _play(tool, name="どこにもない名前", kind=kind)
    assert seen["k"] == want
