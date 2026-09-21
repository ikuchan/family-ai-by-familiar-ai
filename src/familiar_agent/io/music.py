"""音楽の口（知-aa 段 1・2026-09-21）。MPRIS（D-Bus）で `spotifyd` を操る。

**状態は溜めない**（`ユースケース④`）。鳴っているか・いまの曲・音量は、必要なときに MPRIS から
読む。持ち続けるのは音楽の意図（O の開いた意図）だけである。

実機で確かめた 2 つの癖（2026-09-21）。**バス名は起動のたびに変わる**
（`org.mpris.MediaPlayer2.spotifyd.instance<PID>`）ので前方一致で探す。**MPRIS の口は、この機が
再生中の機器になってから現れる**ので、無いときは `spotifyd` 自身の口（`rs.spotifyd.instance…` の
`TransferPlayback`）で再生をこちらへ移してから鳴らす。

`Bus` は薄い包み（`dbus-next` のセッションバス）で、差し替えられるように口だけを決めてある。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

MPRIS_PREFIX = "org.mpris.MediaPlayer2."
SPOTIFYD_PREFIX = "rs.spotifyd."
_MPRIS_PATH = "/org/mpris/MediaPlayer2"
_SPOTIFYD_PATH = "/rs/spotifyd/Controls"


async def find_player_name(bus: Any) -> "str | None":
    """MPRIS の口の名前。まだ鳴らしていなければ無い（それは異常ではない）。"""
    names = await bus.list_names()
    for name in names:
        if name.startswith(MPRIS_PREFIX):
            return name
    return None


async def _player(bus: Any) -> Any:
    name = await find_player_name(bus)
    return await bus.player(name) if name else None


async def play(bus: Any, uri: str) -> bool:
    """プレイリストか曲を鳴らし始める。鳴らせたら True。

    MPRIS が無ければ、先に再生をこの機へ移す（`TransferPlayback`）。それでも口が出なければ
    False を返す——**鳴らせなかったことは、そのまま伝える**（作り話をしない）。
    """
    player = await _player(bus)
    if player is None:
        controls = await bus.controls()
        if controls is None:
            logger.info("音楽：spotifyd が居ないので鳴らせない")
            return False
        await controls.call_transfer_playback()
        player = await _player(bus)
        if player is None:
            logger.info("音楽：再生をこちらへ移したが、MPRIS の口が出ない")
            return False
    await player.call_open_uri(uri)
    logger.info("音楽：鳴らし始めた（%s）", uri)
    return True


async def stop(bus: Any) -> bool:
    """止める。鳴っていなければ False。"""
    player = await _player(bus)
    if player is None:
        return False
    await player.call_stop()
    logger.info("音楽：止めた")
    return True


async def next_track(bus: Any) -> bool:
    """次の曲へ。鳴っていなければ False。"""
    player = await _player(bus)
    if player is None:
        return False
    await player.call_next()
    return True


async def set_volume(bus: Any, value: float) -> bool:
    """音量を書く（0.0〜1.0 に収める）。鳴っていなければ False。"""
    player = await _player(bus)
    if player is None:
        return False
    await player.set_volume(max(0.0, min(1.0, float(value))))
    return True


async def status(bus: Any) -> dict:
    """いまの様子を読む。鳴っていなければ空の辞書。"""
    player = await _player(bus)
    if player is None:
        return {}
    state = await player.get_playback_status()
    meta = await player.get_metadata()
    volume = await player.get_volume()
    return {
        "playing": state == "Playing",
        "title": _one(meta.get("xesam:title")),
        "artist": _one(meta.get("xesam:artist")),
        "volume": float(volume),
    }


def _one(variant: Any) -> str:
    """MPRIS の値は包まれていて、演者は並びで来る。先頭だけを文字にする。"""
    if variant is None:
        return ""
    value = getattr(variant, "value", variant)
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value)
