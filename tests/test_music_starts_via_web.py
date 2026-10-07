"""再生は Spotify の Web API で始め、鳴ったものを確かめる（知-al・2026-10-07 実機・知-ak から独立）。

22:01 に「音楽をかけて」で曲（ヨルシカ「だから僕は音楽を辞めた」）を MPRIS の `OpenUri` で開いたが、鳴ったのは Spotify 側に
残っていた前のアルバム（米津玄師「Blue Jasmine」）で、パジュは確かめずに「かけましたよ、ヨルシカの…」と答えた。いまは
`PUT /me/player/play`（曲は `uris`・プレイリスト等は `context_uri`）で始め、0.5 秒おきに最大 3 秒〔仮〕、Spotify 側の「いま
鳴っているもの」が頼んだものと合うかを見る。合わなければ「かけようとしたが、別の曲が鳴っている」と返す（失敗）。
Web API が無い・失敗したら、いまどおり MPRIS の道で鳴らす。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.io import spotify_web
from familiar_agent.tools import music as music_mod
from familiar_agent.tools.music import MusicTool

TABLE = (("ケイマン", "spotify:playlist:aaa", True),)


@pytest.fixture(autouse=True)
def _quick(monkeypatch):
    monkeypatch.setattr(music_mod, "_CONFIRM_SEC", 0.2)
    monkeypatch.setattr(music_mod, "_CONFIRM_TICK", 0.05)


def _tool(now_playing, *, play=None):
    io = MagicMock()
    io.play = AsyncMock(return_value=True)
    io.set_shuffle = AsyncMock(return_value=True)
    web = MagicMock()
    web.play = MagicMock(side_effect=play or (lambda name, uri: True))
    web.now_playing = MagicMock(return_value=now_playing)
    web.search = MagicMock(
        return_value=[
            {"title": "だから僕は音楽を辞めた", "artist": "ヨルシカ", "uri": "spotify:track:yoru"}
        ]
    )
    tool = MusicTool(
        io=io, bus=lambda: MagicMock(), table=lambda: TABLE, web=web, device_name="パジュ"
    )
    return tool, io, web


def _play(tool, **inp):
    return asyncio.run(tool.call("play_music", inp))


def test_a_playlist_starts_through_the_web_and_is_confirmed():
    tool, io, web = _tool(
        {"playing": True, "context": "spotify:playlist:aaa", "item": "spotify:track:x"}
    )
    text, ok = _play(tool, name="ケイマン")
    assert ok and "かけ始めた" in text
    web.play.assert_called_once_with("パジュ", "spotify:playlist:aaa")
    io.play.assert_not_awaited()  # MPRIS の OpenUri は使わない
    io.set_shuffle.assert_awaited_once()  # ランダムはいまどおり MPRIS


def test_a_track_is_confirmed_by_the_item():
    tool, io, web = _tool({"playing": True, "context": None, "item": "spotify:track:yoru"})
    text, ok = _play(tool, name="だから僕は音楽を辞めた")
    assert ok
    web.play.assert_called_once_with("パジュ", "spotify:track:yoru")


def test_todays_mismatch_is_reported_not_claimed():
    """22:01 の形：曲を頼んで、前のアルバムが鳴っている。"""
    tool, io, web = _tool(
        {
            "playing": True,
            "context": "spotify:album:old",
            "item": "spotify:track:6nrImcv9m7yG4DbAxE5j8S",
        }
    )
    text, ok = _play(tool, name="だから僕は音楽を辞めた")
    assert not ok
    assert "別の曲が鳴っている" in text


def test_without_the_web_mpris_is_used():
    io = MagicMock()
    io.play = AsyncMock(return_value=True)
    io.set_shuffle = AsyncMock(return_value=True)
    tool = MusicTool(io=io, bus=lambda: MagicMock(), table=lambda: TABLE)
    _, ok = _play(tool, name="ケイマン")
    assert ok and io.play.await_args.args[1] == "spotify:playlist:aaa"


def test_a_web_failure_falls_back_to_mpris():
    def boom(name, uri):
        raise RuntimeError("502")

    tool, io, web = _tool({}, play=boom)
    _, ok = _play(tool, name="ケイマン")
    assert ok and io.play.await_args.args[1] == "spotify:playlist:aaa"


# ── Web API の口 ──────────────────────────────────────────────────────────────


def _web(responses):
    calls = []

    def http(method, url, *, token, body=None):
        calls.append((method, url, body))
        return responses.get((method, url.split("?")[0].rsplit("/v1", 1)[-1]), {})

    sp = spotify_web.Spotify(token_path="/nonexistent", http=http)
    sp._token = lambda: {"access_token": "t"}  # type: ignore[method-assign]
    return sp, calls


def test_play_uses_uris_for_a_track_and_context_for_the_rest():
    devices = {("GET", "/me/player/devices"): {"devices": [{"name": "パジュ", "id": "dev1"}]}}
    sp, calls = _web(devices)
    assert sp.play("パジュ", "spotify:track:t1") is True
    assert sp.play("パジュ", "spotify:playlist:p1") is True
    puts = [c for c in calls if c[0] == "PUT"]
    assert puts[0][1].endswith("/me/player/play?device_id=dev1") and puts[0][2] == {
        "uris": ["spotify:track:t1"]
    }
    assert puts[1][2] == {"context_uri": "spotify:playlist:p1"}


def test_play_without_the_device_is_false():
    sp, calls = _web({("GET", "/me/player/devices"): {"devices": []}})
    assert sp.play("パジュ", "spotify:track:t1") is False
    assert not [c for c in calls if c[0] == "PUT"]


def test_now_playing_reads_the_device_item_and_context():
    sp, _ = _web(
        {
            ("GET", "/me/player"): {
                "is_playing": True,
                "device": {"name": "パジュ"},
                "item": {"uri": "spotify:track:t1"},
                "context": {"uri": "spotify:album:a1"},
            }
        }
    )
    assert sp.now_playing() == {
        "device": "パジュ",
        "playing": True,
        "item": "spotify:track:t1",
        "context": "spotify:album:a1",
    }
