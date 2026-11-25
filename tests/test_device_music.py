"""端-c c6b 段 1：端末の求めの音楽は、頼んだ端末の Spotify を開いて示す（課題9 b2・§8.9）。

- 探し方は家と同じ（`MusicTool.find`）。家では鳴らさない（spotifyd も家の機器も、鳴っている印も音量も触らない）。
- 端末へ `music`（Spotify の URI）を送る。名前が無ければ URI を空にして、端末の Spotify を開くだけにする。
- 機体は端末で鳴ったかを知らないので、返りは「端末の Spotify で開く」と書く（かけた、とは書かない）。1 つの求めで 1 回まで。
- 止める・次の曲・音量・すすめた曲の返事は、端末の求めではいままでどおり使わない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.loop.event_loop import InformationProcessing

pytestmark = pytest.mark.real_window

TAB = "tablet-1"
URI = "spotify:playlist:0000000000000000000000"


def _ip(device: str = TAB):
    a = MagicMock()
    a.config.agent_names = ["パジュ"]
    a.config.tts_gain = 1.0
    a._pmm.presence_status = MagicMock(return_value=[])
    a._music_tool.call = AsyncMock(return_value=("かけ始めた", True))
    a._music_tool.find = AsyncMock(return_value=("あなたのプレイリスト「朝」を", URI, True))
    ip = InformationProcessing(a)
    ip._dif = MagicMock()
    ip._req.trigger_kind = "発話"
    ip._req.device = device
    got: list = []
    ip.set_device_listener(lambda dev, kind, payload: got.append((dev, kind, payload)))
    return ip, got


def _run(ip, action, inp):
    asyncio.run(ip._run_lookup_body(action, inp, "q", None))
    return ip._triggers.get_nowait()


def test_device_music_is_found_as_at_home_and_opened_on_the_device():
    ip, got = _ip()
    done = _run(ip, "play_music", {"name": "朝", "order": "ランダム"})
    ip._agent._music_tool.find.assert_awaited_once_with({"name": "朝", "order": "ランダム"})
    ip._agent._music_tool.call.assert_not_called()  # 家では鳴らさない
    assert got == [(TAB, "music", {"uri": URI})]
    assert not done.failed
    assert "端末の Spotify" in done.result and "かけ始めた" not in done.result


def test_device_music_not_found_is_told_back_without_sending():
    ip, got = _ip()
    ip._agent._music_tool.find = AsyncMock(
        return_value="「朝」は見つからなかったので、かけられない"
    )
    done = _run(ip, "play_music", {"name": "朝"})
    assert got == [] and done.failed and "見つからなかった" in done.result


@pytest.mark.parametrize("name", ["", "音楽"])
def test_device_music_without_a_name_only_opens_spotify(name):
    ip, got = _ip()
    done = _run(ip, "play_music", {"name": name})
    ip._agent._music_tool.find.assert_not_called()
    ip._agent._music_tool.call.assert_not_called()  # 家の続きもかけない
    assert got == [(TAB, "music", {"uri": ""})] and not done.failed


def test_device_music_is_sent_only_once_per_request():
    ip, got = _ip()
    _run(ip, "play_music", {"name": "朝"})
    again = _run(ip, "play_music", {"name": "朝"})
    assert len(got) == 1 and again.failed


def test_home_music_still_plays_at_home():
    """**反証の側**：家の求めは、いままでどおり家で鳴らし、端末へは送らない。"""
    ip, got = _ip(device="")
    _run(ip, "play_music", {"name": "朝"})
    ip._agent._music_tool.call.assert_awaited_once()
    assert got == []


@pytest.mark.parametrize(
    "action", ["stop_music", "next_track", "music_volume", "music_suggestion_reply"]
)
def test_other_music_tools_are_still_refused_for_a_device(action):
    ip, got = _ip()
    done = _run(ip, action, {})
    assert done.failed and got == []
    ip._agent._music_tool.call.assert_not_called()


def test_what_the_device_spotify_cannot_open_is_not_sent():
    ip, got = _ip()
    ip._agent._music_tool.find = AsyncMock(
        return_value=("「朝」を", "https://example.com/x", False)
    )
    done = _run(ip, "play_music", {"name": "朝"})
    assert got == [] and done.failed


def test_a_device_request_is_not_shown_the_home_music():
    """家で鳴っていても、端末の求めの主LLM には家の `[音楽]` を見せない（「もう鳴っている」と端末の頼みを断らない）。"""
    ip, _ = _ip()
    ip._agent._music_state = MagicMock()
    ip._music_status = AsyncMock(return_value={"playing": True, "title": "朝"})
    assert asyncio.run(ip._music_now()) == ""
    assert asyncio.run(ip._no_music()) is None
    ip._agent._music_tool.frame = AsyncMock(return_value="[音楽] 鳴っている")
    ip._req.device = ""
    assert asyncio.run(ip._music_now()) == "[音楽] 鳴っている"  # 家の求めには見せる


@pytest.mark.parametrize("uri", ["spotify:", "spotify:track:\x85", "spotify:track:a b"])
def test_the_machine_takes_the_same_spotify_uris_as_the_app(uri):
    """アプリ（`Wire.isSpotify`）と同じ決まり：`spotify:` のあとに空白の無い ASCII の字が 1 字以上。"""
    ip, got = _ip()
    ip._agent._music_tool.find = AsyncMock(return_value=("「朝」を", uri, False))
    assert _run(ip, "play_music", {"name": "朝"}).failed and got == []


def test_opening_on_the_device_is_told_lightly_and_not_kept_silent():
    """家では鳴った音で分かるので黙るが、端末では何を開いたかと、再生は端末で押すことを伝える。"""
    from familiar_agent.core.completion_kind import kind_of

    ip, _ = _ip()
    done = _run(ip, "play_music", {"name": "朝", "order": "ランダム"})
    assert "端末の Spotify のまま" in done.result
    got = kind_of("play_music", failed=False, result=done.result, origin="発話")
    assert got is not None and got[1] == ("tell_light",)
    home = kind_of("play_music", failed=False, result="「朝」をかけ始めた", origin="発話")
    assert home is not None and home[1] == ("silent",)
