"""プレイリストの中身を晩に読んで DB に持つ（知-aa 段 3 の土台・2026-10-01・本人の決定ア）。

古い口（`/playlists/{id}/tracks`）は自分で作ったプレイリストでも 403 だった。新しい口（`/playlists/{id}/items`）は
読めた（実機 2026-10-01・「ケイマン」85 曲）。中身は晩に読み直して `agent_state` の `music_catalog` に置き、会話の
ときは DB を引くだけにする（頼まれるたびに数十回 Spotify を呼ぶと、鳴るまで待たせる）。読めなかったプレイリストは
前の晩の中身を残す。Spotify を呼ぶのは `io/spotify_web.Spotify` の 1 か所だけで、試験では `http` を差し替える。
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from familiar_agent.core import music_catalog as mc
from familiar_agent.io.spotify_web import API, Spotify


def _spotify(pages: dict) -> Spotify:
    """`pages[path]` を返す偽の Web API（ページのたどりは `next` の URL で）。無い道は失敗（空の辞書）。"""
    calls: list[str] = []

    def http(method, url, *, token, body=None):
        path = url[len(API) :]
        calls.append(path)
        return pages.get(path, {})

    sp = Spotify(token_path="/nonexistent", http=http)
    sp._token = lambda: {"access_token": "x"}  # type: ignore[method-assign]
    sp.calls = calls  # type: ignore[attr-defined]
    return sp


def _track(title, artist, uri):
    return {"name": title, "uri": uri, "artists": [{"name": artist}]}


def test_my_playlists_follow_the_pages():
    sp = _spotify(
        {
            "/me": {"id": "me"},
            "/me/playlists?limit=50": {
                "items": [
                    {
                        "id": "p1",
                        "name": "ケイマン",
                        "uri": "spotify:playlist:p1",
                        "owner": {"id": "me"},
                    }
                ],
                "next": f"{API}/me/playlists?offset=50&limit=50",
            },
            "/me/playlists?offset=50&limit=50": {
                "items": [
                    {"id": "p2", "name": "人の", "uri": "spotify:playlist:p2", "owner": {"id": "x"}}
                ],
                "next": None,
            },
        }
    )
    got = sp.my_playlists()
    assert [(p["name"], p["mine"]) for p in got] == [("ケイマン", True), ("人の", False)]


def test_playlist_items_use_the_new_way_and_read_both_shapes():
    sp = _spotify(
        {
            "/playlists/p1/items?limit=100": {
                "items": [
                    {"item": _track("Blue Jasmine", "A", "spotify:track:1")},
                    {"track": _track("Tokyo", "B", "spotify:track:2")},
                    {"item": None},
                ],
                "next": None,
            }
        }
    )
    assert sp.playlist_items("p1") == [
        {"title": "Blue Jasmine", "artist": "A", "uri": "spotify:track:1"},
        {"title": "Tokyo", "artist": "B", "uri": "spotify:track:2"},
    ]
    assert all("/tracks" not in c for c in sp.calls)  # 古い口は使わない


def test_a_playlist_that_cannot_be_read_is_none():
    assert _spotify({}).playlist_items("p1") is None


def test_saved_albums_and_tracks_and_search():
    sp = _spotify(
        {
            "/me/albums?limit=50": {
                "items": [
                    {
                        "album": {
                            "name": "STRAY SHEEP",
                            "uri": "spotify:album:a1",
                            "artists": [{"name": "米津玄師"}],
                        }
                    }
                ],
                "next": None,
            },
            "/me/tracks?limit=50": {
                "items": [{"track": _track("Lemon", "米津玄師", "spotify:track:l")}],
                "next": None,
            },
            "/search?q=Lemon&type=track&limit=5&market=JP": {
                "tracks": {"items": [_track("Lemon", "米津玄師", "spotify:track:l")]}
            },
        }
    )
    assert sp.saved_albums() == [
        {"title": "STRAY SHEEP", "artist": "米津玄師", "uri": "spotify:album:a1"}
    ]
    assert sp.saved_tracks() == [{"title": "Lemon", "artist": "米津玄師", "uri": "spotify:track:l"}]
    assert sp.search("Lemon", "track") == [
        {"title": "Lemon", "artist": "米津玄師", "uri": "spotify:track:l"}
    ]


# ── 目録（DB）と晩の読み直し ───────────────────────────────────────────────


@pytest.fixture
def clean_catalog():
    mc.clear()
    yield
    mc.clear()


def test_the_catalog_round_trips(clean_catalog):
    cat = mc.Catalog(
        playlists=[
            mc.Playlist(
                "p1",
                "ケイマン",
                "spotify:playlist:p1",
                [{"title": "Tokyo", "artist": "B", "uri": "u"}],
            )
        ],
        albums=[],
        tracks=[],
    )
    assert mc.store(cat)
    got = mc.stored()
    assert got.playlists[0].name == "ケイマン" and got.playlists[0].tracks[0]["title"] == "Tokyo"


def test_the_night_refresh_keeps_what_it_could_not_read(clean_catalog):
    from familiar_agent.loop.rest_music import refresh_catalog

    mc.store(
        mc.Catalog(
            playlists=[
                mc.Playlist(
                    "p2",
                    "人の",
                    "spotify:playlist:p2",
                    [{"title": "前の曲", "artist": "C", "uri": "u2"}],
                )
            ],
            albums=[],
            tracks=[],
        )
    )
    sp = _spotify(
        {
            "/me": {"id": "me"},
            "/me/playlists?limit=50": {
                "items": [
                    {
                        "id": "p1",
                        "name": "ケイマン",
                        "uri": "spotify:playlist:p1",
                        "owner": {"id": "me"},
                    },
                    {
                        "id": "p2",
                        "name": "人の",
                        "uri": "spotify:playlist:p2",
                        "owner": {"id": "x"},
                    },
                ],
                "next": None,
            },
            "/playlists/p1/items?limit=100": {
                "items": [{"item": _track("Tokyo", "B", "u1")}],
                "next": None,
            },
            "/me/albums?limit=50": {"items": [], "next": None},
            "/me/tracks?limit=50": {"items": [], "next": None},
        }
    )
    agent = MagicMock()
    agent._music_tool._web = sp
    r = asyncio.run(refresh_catalog(agent))
    got = {p.name: p for p in mc.stored().playlists}
    assert [t["title"] for t in got["ケイマン"].tracks] == ["Tokyo"]
    assert [t["title"] for t in got["人の"].tracks] == ["前の曲"]  # 読めなかったので前の中身
    assert r.playlists == 2 and r.unread == 1


def test_no_spotify_means_nothing_to_do(clean_catalog):
    from familiar_agent.loop.rest_music import refresh_catalog

    agent = MagicMock()
    agent._music_tool = None
    r = asyncio.run(refresh_catalog(agent))
    assert r.playlists == 0 and mc.stored().playlists == []
