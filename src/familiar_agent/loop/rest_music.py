"""REST で音楽の目録を読み直す（知-aa 段 3・2026-10-01・本人の決定ア）。

晩に、自分の Spotify のプレイリストの中身と、保存したアルバムと曲を読み直して DB に置く（`core/music_catalog`）。
読めなかったプレイリストは前の晩の中身を残す（Spotify は落ちる前提のもの）。音楽の器が無い機体では何もしない。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from ..core import music_catalog as mc

logger = logging.getLogger(__name__)


@dataclass
class CatalogResult:
    playlists: int = 0
    unread: int = 0  # 中身を読めず、前の晩の中身を残したプレイリスト
    tracks: int = 0


async def refresh_catalog(agent) -> CatalogResult:
    """目録を読み直す。Spotify を呼ぶのはスレッドで（ネットワークの待ちでループを止めない）。"""
    tool = getattr(agent, "_music_tool", None)
    web = getattr(tool, "_web", None) if tool is not None else None
    if web is None:
        return CatalogResult()
    return await asyncio.to_thread(_refresh, web)


def _refresh(web) -> CatalogResult:
    before = {p.id: p for p in mc.stored().playlists}
    result = CatalogResult()
    playlists: list[mc.Playlist] = []
    for p in web.my_playlists():
        items = web.playlist_items(p["id"])
        if items is None:
            result.unread += 1
            items = before[p["id"]].tracks if p["id"] in before else []
        playlists.append(mc.Playlist(p["id"], p["name"], p["uri"], list(items), p["mine"]))
        result.tracks += len(items)
    result.playlists = len(playlists)
    if playlists or not before:
        mc.store(
            mc.Catalog(playlists=playlists, albums=web.saved_albums(), tracks=web.saved_tracks())
        )
    logger.info(
        "rest 音楽の目録：プレイリスト %d・曲 %d（読めず前のまま %d）",
        result.playlists,
        result.tracks,
        result.unread,
    )
    return result
