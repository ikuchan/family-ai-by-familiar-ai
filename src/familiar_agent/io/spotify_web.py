"""Spotify Web API の薄い口（知-aa・2026-09-21）。

持つのは**鍵の更新**（アクセス鍵は 1 時間で切れる）と、**機器の切り替え**（`PUT /v1/me/player`・
`play: false` なので音は鳴らない）と、**読むだけの口**（知-aa 段 3・2026-10-01：自分のプレイリストの一覧と
中身、保存したアルバムと曲、検索）。Spotify を呼ぶのはこのクラスだけにする。

プレイリストの中身は**新しい口 `/playlists/{id}/items`** で読む。古い口 `/playlists/{id}/tracks` は、このアプリ
では自分で作ったプレイリストでも 403 だった（実機 2026-10-01）。ジャンル・関連アーティスト・おすすめは使えない
（ジャンルの欄が無い・403・404）。

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

    def play(self, name: str, uri: str) -> bool:
        """機器 `name` で `uri` を鳴らし始める（知-al・2026-10-07）。機器が見つからなければ False。

        曲は `uris`、プレイリスト・アルバム・アーティストは `context_uri`。MPRIS の `OpenUri` では曲の URI に
        切り替わらず、Spotify 側に残っていた前のアルバムが鳴った（実機 22:01）。機器の切り替えも兼ねる。
        """
        device = self.device_id(name or "")
        if not device:
            return False
        body: dict = {"uris": [uri]} if ":track:" in uri else {"context_uri": uri}
        self._call("PUT", f"/me/player/play?device_id={device}", body)
        logger.info("音楽：機器「%s」で鳴らし始めた（%s）", name, uri)
        return True

    def resume(self, name: str) -> bool:
        """機器 `name` で、止まっていた続きを鳴らす（曲を指定しない・出-ay 段 4-4e）。機器が見つからなければ False。

        Spotify 側に続きが無くても、ここは失敗を返さない（`_http` が誤りを空で返す）。鳴ったかは呼び手が確かめる。
        """
        device = self.device_id(name or "")
        if not device:
            return False
        self._call("PUT", f"/me/player/play?device_id={device}", {})
        logger.info("音楽：機器「%s」で止まっていた続きを頼んだ", name)
        return True

    def now_playing(self) -> dict:
        """いま鳴っているもの（機器名・鳴っているか・曲の URI・その上の URI）。読めなければ空の値。"""
        st = self._call("GET", "/me/player") or {}
        return {
            "device": (st.get("device") or {}).get("name"),
            "playing": bool(st.get("is_playing")),
            "item": (st.get("item") or {}).get("uri"),
            "context": (st.get("context") or {}).get("uri"),
        }

    # ── 読むだけの口（知-aa 段 3）────────────────────────────────────────────

    def _pages(self, path: str) -> "list[dict] | None":
        """`next` をたどって `items` を集める。最初のページが読めなければ None。"""
        out: list[dict] = []
        first = True
        while path:
            got = self._call("GET", path)
            if "items" not in got:
                return None if first else out
            out += [i for i in got.get("items") or [] if i]
            nxt = str(got.get("next") or "")
            path = nxt[len(API) :] if nxt.startswith(API) else ""
            first = False
        return out

    def my_playlists(self) -> "list[dict]":
        """自分のプレイリスト（持ち主が自分か `mine` を添える）。読めなければ空。"""
        me = str(self._call("GET", "/me").get("id") or "")
        return [
            {
                "id": str(p.get("id") or ""),
                "name": str(p.get("name") or ""),
                "uri": str(p.get("uri") or ""),
                "mine": bool(me) and str((p.get("owner") or {}).get("id") or "") == me,
            }
            for p in self._pages("/me/playlists?limit=50") or []
        ]

    def playlist_items(self, playlist_id: str) -> "list[dict] | None":
        """プレイリストの曲（新しい口）。読めなければ None（呼び手は前の中身を残す）。"""
        items = self._pages(f"/playlists/{playlist_id}/items?limit=100")
        if items is None:
            return None
        return [t for t in (_track_of(i.get("item") or i.get("track")) for i in items) if t]

    def saved_albums(self) -> "list[dict]":
        return [
            t
            for t in (_track_of(i.get("album")) for i in self._pages("/me/albums?limit=50") or [])
            if t
        ]

    def saved_tracks(self) -> "list[dict]":
        return [
            t
            for t in (_track_of(i.get("track")) for i in self._pages("/me/tracks?limit=50") or [])
            if t
        ]

    def search(self, query: str, kind: str, *, limit: int = 5) -> "list[dict]":
        """Spotify 全体の検索。`kind` は track・artist・album・playlist。"""
        q = urllib.parse.urlencode({"q": query, "type": kind, "limit": limit, "market": "JP"})
        got = self._call("GET", f"/search?{q}").get(f"{kind}s") or {}
        return [t for t in (_track_of(i) for i in got.get("items") or []) if t]


def _track_of(obj: "dict | None") -> "dict | None":
    """曲・アルバム・アーティスト・プレイリストを {title, artist, uri} にそろえる。無ければ None。"""
    if not isinstance(obj, dict) or not obj.get("uri"):
        return None
    artists = obj.get("artists") or []
    artist = str(artists[0].get("name") or "") if artists and isinstance(artists[0], dict) else ""
    owner = obj.get("owner") or {}
    if not artist and isinstance(owner, dict):
        artist = str(owner.get("display_name") or "")
    return {"title": str(obj.get("name") or ""), "artist": artist, "uri": str(obj["uri"])}
