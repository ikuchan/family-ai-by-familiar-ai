"""Spotify Web API の薄い口（知-aa・2026-09-21）。

使うのは 2 つだけである。**鍵の更新**（アクセス鍵は 1 時間で切れる）と、**機器の切り替え**
（`PUT /v1/me/player`・`play: false` なので音は鳴らない）。

なぜ要るか。**MPRIS の口は、この機が再生中の機器になってから現れる**（実機 2026-09-21）。
起動しただけでは口が無く、鳴らそうとしても掴めない。そこで**鳴らす直前**にここを通して機器を
こちらへ切り替える（本人の決定：起動時や REST では切り替えない——家族が別の機器で聴いている
のを奪わないため。頼まれた瞬間だけ持ってくる）。

曲を選ぶのも鳴らすのも MPRIS（`io/music.py`）の仕事で、ここではやらない。鍵は
`~/.familiar_ai/spotify_token.json`（本人のみ読める）。`client_id` は `.env`（秘密鍵は要らない・PKCE）。
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

API = "https://api.spotify.com/v1"
TOKEN_URL = "https://accounts.spotify.com/api/token"
DEFAULT_TOKEN_PATH = str(Path.home() / ".familiar_ai" / "spotify_token.json")


class Unauthorized(Exception):
    """鍵の期限切れ（401）。更新して 1 度だけやり直す。"""


def _http(method: str, url: str, *, token: str, body: "dict | None" = None) -> dict:
    req = urllib.request.Request(
        url,
        method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        data=json.dumps(body).encode() if body is not None else None,
    )
    try:
        raw = urllib.request.urlopen(req, timeout=10).read()
        return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise Unauthorized() from e
        logger.info("音楽：Web API が %s を返した（%s）", e.code, url.rsplit("/", 1)[-1])
        return {}


def _refresh(refresh_token: str, *, client_id: str, token_path: str) -> str:
    """鍵を更新して書き戻す。新しいアクセス鍵を返す。"""
    data = urllib.parse.urlencode(
        {"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": client_id}
    ).encode()
    req = urllib.request.Request(
        TOKEN_URL, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    new = json.loads(urllib.request.urlopen(req, timeout=10).read())
    path = Path(token_path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved.update(new)
    path.write_text(json.dumps(saved), encoding="utf-8")
    path.chmod(0o600)
    return str(new["access_token"])


class Spotify:
    """鍵の更新と機器の切り替えだけを持つ。無ければ静かに False を返す（鳴らす側が断る）。"""

    def __init__(
        self,
        *,
        token_path: str = DEFAULT_TOKEN_PATH,
        client_id: str = "",
        http: Any = _http,
        refresh: Any = None,
    ) -> None:
        self._path = token_path
        self._client_id = client_id or os.environ.get("SPOTIFY_CLIENT_ID", "")
        self._http = http
        self._refresh = refresh or (
            lambda r: _refresh(r, client_id=self._client_id, token_path=self._path)
        )

    def _token(self) -> dict:
        try:
            return json.loads(Path(self._path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}

    def _call(self, method: str, path: str, body: "dict | None" = None) -> dict:
        tok = self._token()
        access = str(tok.get("access_token") or "")
        if not access:
            return {}
        try:
            return self._http(method, f"{API}{path}", token=access, body=body)
        except Unauthorized:
            # 期限切れ。**1 度だけ**更新してやり直す（無限に繰り返さない）。
            refresh_token = str(tok.get("refresh_token") or "")
            if not refresh_token:
                return {}
            access = self._refresh(refresh_token)
            try:
                return self._http(method, f"{API}{path}", token=access, body=body)
            except Unauthorized:
                logger.info("音楽：鍵を更新しても通らない（もう一度承認が要る）")
                return {}

    def device_id(self, name: str) -> "str | None":
        """機器の名前から id。見つからなければ None。"""
        if not name:
            return None
        for d in self._call("GET", "/me/player/devices").get("devices", []) or []:
            if name in str(d.get("name") or ""):
                return str(d.get("id"))
        return None

    def activate(self, name: str) -> bool:
        """その機器を現役にする（**音は鳴らさない**）。切り替えられたら True。

        頼まれた瞬間に呼ぶ。別の機器で鳴っていればこちらへ移ることになるが、それは
        「かけて」と言われた場面なので意図どおりである。
        """
        device = self.device_id(name or "")
        if not device:
            return False
        self._call("PUT", "/me/player", {"device_ids": [device], "play": False})
        logger.info("音楽：機器「%s」へ切り替えた（音は鳴らさない）", name)
        return True
