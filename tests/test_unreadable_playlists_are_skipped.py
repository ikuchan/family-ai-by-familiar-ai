"""中身を読めない他人のプレイリストは、覚えて毎晩は読まない（知-ao・2026-10-10・本人の決定ア）。

Spotify のプレイリストの中身を読む口は、自分が持っているか共同編集者になっているものだけで、ほかは 403 を返す（公式の
リファレンス）。家族のアカウントの 3 つと、uDiscover・Tom Bravo の 2 つが毎晩 403 になり、目録では最初から 0 曲だった。
他人のもので 1 度読めなかったら、その日を `unreadable_since` に覚え、7 日〔仮〕たつまで読みに行かない。7 日たったら 1 度
読み直し（共同編集者が足されたなどを拾う）、読めたら印を消す。自分のものが読めないのは一時の失敗として、いままでどおり
前の中身を残して毎晩読む。
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest

from familiar_agent.core import music_catalog as mc
from familiar_agent.loop import rest_music

TODAY = "2026-10-10"


@pytest.fixture
def catalog(monkeypatch):
    data: dict = {"c": mc.Catalog()}
    monkeypatch.setattr(mc, "stored", lambda: data["c"])
    monkeypatch.setattr(mc, "store", lambda c: data.__setitem__("c", c) or True)
    return data


def _web(playlists, items):
    """`items` はプレイリスト ID → 曲の並び（None なら 403）。"""
    web = MagicMock()
    web.my_playlists = MagicMock(return_value=playlists)
    web.playlist_items = MagicMock(side_effect=lambda pid: items.get(pid))
    web.saved_albums = MagicMock(return_value=[])
    web.saved_tracks = MagicMock(return_value=[])
    return web


TORA = {"id": "tora", "name": "トラ", "uri": "spotify:playlist:tora", "mine": False}
KEIMAN = {"id": "keiman", "name": "ケイマン", "uri": "spotify:playlist:keiman", "mine": True}
SONG = [{"title": "曲", "artist": "人", "uri": "spotify:track:1"}]


def _by_id(cat):
    return {p.id: p for p in cat.playlists}


def test_an_unreadable_playlist_of_someone_else_is_marked(catalog, caplog):
    web = _web([TORA, KEIMAN], {"tora": None, "keiman": SONG})
    with caplog.at_level(logging.INFO):
        rest_music._refresh(web, today=TODAY)
    assert _by_id(catalog["c"])["tora"].unreadable_since == TODAY
    assert _by_id(catalog["c"])["keiman"].unreadable_since == ""
    assert "中身を読めない他人のもの 1" in caplog.text


def test_a_marked_playlist_is_not_read_for_seven_days(catalog):
    catalog["c"] = mc.Catalog(
        playlists=[mc.Playlist("tora", "トラ", "spotify:playlist:tora", [], False, "2026-10-04")]
    )
    web = _web([TORA], {"tora": SONG})
    rest_music._refresh(web, today=TODAY)  # 6 日目
    web.playlist_items.assert_not_called()
    assert _by_id(catalog["c"])["tora"].unreadable_since == "2026-10-04"


def test_after_seven_days_it_is_read_again_and_the_mark_goes(catalog):
    catalog["c"] = mc.Catalog(
        playlists=[mc.Playlist("tora", "トラ", "spotify:playlist:tora", [], False, "2026-10-03")]
    )
    web = _web([TORA], {"tora": SONG})
    rest_music._refresh(web, today=TODAY)  # 7 日目
    web.playlist_items.assert_called_once_with("tora")
    tora = _by_id(catalog["c"])["tora"]
    assert tora.unreadable_since == "" and tora.tracks == SONG


def test_my_own_unreadable_playlist_keeps_the_old_contents_and_no_mark(catalog):
    catalog["c"] = mc.Catalog(
        playlists=[mc.Playlist("keiman", "ケイマン", "spotify:playlist:keiman", SONG, True)]
    )
    web = _web([KEIMAN], {"keiman": None})
    rest_music._refresh(web, today=TODAY)
    keiman = _by_id(catalog["c"])["keiman"]
    assert keiman.unreadable_since == "" and keiman.tracks == SONG


def test_the_mark_survives_a_round_trip():
    cat = mc.Catalog(
        playlists=[mc.Playlist("tora", "トラ", "spotify:playlist:tora", [], False, TODAY)]
    )
    back = mc._from_json(json.loads(json.dumps(mc._to_json(cat))))
    assert back.playlists[0].unreadable_since == TODAY
