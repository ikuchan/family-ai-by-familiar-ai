"""Spotify Web API の薄い口（知-aa・2026-09-21）。

使うのは 2 つだけ——**鍵の更新**（アクセス鍵は 1 時間で切れる）と、**機器の切り替え**
（`PUT /v1/me/player`・`play: false`）。MPRIS の口は、この機が再生中の機器になってから現れるので、
鳴らす直前にここを通す。曲を選ぶのも鳴らすのも MPRIS の仕事で、ここではやらない。
"""

from __future__ import annotations

import json

import pytest

from familiar_agent.io import spotify_web as web


class _Http:
    """呼ばれた先を覚える偽の網。401 を 1 度だけ返す設定もできる。"""

    def __init__(self, *, unauthorized_once: bool = False, devices=None) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []
        self._unauthorized = unauthorized_once
        self._devices = (
            devices
            if devices is not None
            else [
                {"id": "d1", "name": "パジュ", "is_active": False},
                {"id": "d2", "name": "Amazon FireTV Stick 4K", "is_active": True},
            ]
        )

    def __call__(self, method: str, url: str, *, token: str, body=None):
        self.calls.append((method, url, body))
        if self._unauthorized and "/me/player" in url:
            self._unauthorized = False
            raise web.Unauthorized()
        if url.endswith("/me/player/devices"):
            return {"devices": self._devices}
        return {}


def _token_file(tmp_path, expires_in: int = 3600) -> str:
    p = tmp_path / "spotify_token.json"
    p.write_text(json.dumps({"access_token": "A", "refresh_token": "R", "expires_in": expires_in}))
    return str(p)


def test_the_device_is_found_by_name(tmp_path):
    http = _Http()
    s = web.Spotify(
        token_path=_token_file(tmp_path), client_id="c", http=http, refresh=lambda r: "A2"
    )
    assert s.device_id("パジュ") == "d1"
    assert s.device_id("知らない機器") is None


def test_activating_switches_without_playing(tmp_path):
    http = _Http()
    s = web.Spotify(
        token_path=_token_file(tmp_path), client_id="c", http=http, refresh=lambda r: "A2"
    )
    assert s.activate("パジュ") is True
    method, url, body = http.calls[-1]
    assert method == "PUT" and url.endswith("/v1/me/player")
    assert body == {"device_ids": ["d1"], "play": False}, "音は鳴らさない"


def test_an_expired_key_is_refreshed_once(tmp_path):
    http = _Http(unauthorized_once=True)
    refreshed: list[str] = []

    def refresh(r):
        refreshed.append(r)
        return "A2"

    s = web.Spotify(token_path=_token_file(tmp_path), client_id="c", http=http, refresh=refresh)
    assert s.activate("パジュ") is True
    assert refreshed == ["R"], "更新は 1 度だけ（古い鍵を使い回さない）"


def test_no_token_file_is_not_an_error(tmp_path):
    s = web.Spotify(
        token_path=str(tmp_path / "無い.json"), client_id="c", http=_Http(), refresh=lambda r: "A"
    )
    assert s.device_id("パジュ") is None
    assert s.activate("パジュ") is False


def test_a_missing_device_is_reported_plainly(tmp_path):
    http = _Http(devices=[{"id": "d2", "name": "FireTV", "is_active": True}])
    s = web.Spotify(
        token_path=_token_file(tmp_path), client_id="c", http=http, refresh=lambda r: "A"
    )
    assert s.activate("パジュ") is False


@pytest.mark.parametrize("name", ["", None])
def test_an_empty_name_never_switches(tmp_path, name):
    http = _Http()
    s = web.Spotify(
        token_path=_token_file(tmp_path), client_id="c", http=http, refresh=lambda r: "A"
    )
    assert s.activate(name) is False
