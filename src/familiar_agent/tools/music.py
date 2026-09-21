"""音楽の道具（知-aa 段 1・2026-09-21・`ユースケース④_音楽再生`）。

主LLM が呼ぶ 4 本——`play_music`（かける）・`stop_music`（止める）・`next_track`（次へ）・
`music_volume`（音量）。鳴らす先は `spotifyd`（MPRIS・`io/music`）で、**状態は溜めず、必要な
ときに読む**（鳴っているか・いまの曲・音量）。持ち続けるのは音楽の意図だけである。

プレイリストは人が書く `MUSIC.md`（`名前：URI[：ランダム]`）から当てる。順番かランダムかは、
**表の 3 つ目の欄が既定で、言われた言葉がそのつど上書きする**（本人の決定・2026-09-21）。
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable
from typing import Any

from ..core import music_rules

logger = logging.getLogger(__name__)

#: 音量を言葉で動かす刻み。
VOLUME_STEP = 0.15

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "play_music",
        "description": (
            "音楽をかける（「音楽かけて」「ケイマンかけて」）。name は覚えているプレイリストの名前。"
            "order に「ランダム」か「順番」を渡すと、そのときだけ順番を変えられる。"
            "鳴っているあいだは、音楽の話だけを聞くようになる。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "プレイリストの名前"},
                "order": {"type": "string", "description": "「ランダム」か「順番」（省略可）"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "stop_music",
        "description": "音楽を止める（「止めて」「音楽やめて」）。止めるとふだんの状態に戻る。",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "next_track",
        "description": "次の曲へ飛ばす（「次の曲」「スキップ」）。",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "music_volume",
        "description": "音楽の音量を変える（「もっと大きく」「小さくして」）。how に「大きく」か「小さく」。",
        "input_schema": {
            "type": "object",
            "properties": {"how": {"type": "string", "description": "「大きく」か「小さく」"}},
            "required": ["how"],
        },
    },
]


class MusicTool:
    """4 本の道具。返りは**そのまま伝えられる文**にする（できなかったことは断りで返す）。"""

    def __init__(
        self,
        *,
        io: Any,
        bus: Callable[[], Any],
        table: Callable[[], "tuple[tuple[str, str, bool], ...]"],
        state: Any = None,
        now: "Callable[[], float] | None" = None,
        web: Any = None,
        device_name: str = "",
    ) -> None:
        self._io = io
        self._bus = bus
        self._table = table
        # 鳴らし始めた時刻と印（寿命と門が見る）。渡されなければ持たない（試験の土台だけの呼び方）。
        self._state = state
        self._now = now or time.time
        # 鳴らす直前に機器をこちらへ切り替える口（知-aa・`io/spotify_web`）。無ければ通さない。
        self._web = web
        self._device_name = device_name

    def get_tool_definitions(self) -> list[dict]:
        return [dict(d) for d in TOOL_DEFINITIONS]

    async def frame(self) -> str:
        """`[音楽]` の枠。鳴っていなければ空。"""
        s = await self._io.status(self._bus())
        if not s or not s.get("playing"):
            return ""
        title = s.get("title") or "曲"
        artist = s.get("artist") or ""
        who = f"／{artist}" if artist else ""
        return f"[音楽] {title}{who}（音量 {int(float(s.get('volume', 0)) * 100)}%）"

    async def call(self, name: str, tool_input: dict) -> "tuple[str, bool]":
        if name == "play_music":
            return await self._play(tool_input)
        if name == "stop_music":
            ok = await self._io.stop(self._bus())
            if ok:
                self._mark(False)
            return ("音楽を止めた", True) if ok else ("いま音楽は鳴っていない", False)
        if name == "next_track":
            ok = await self._io.next_track(self._bus())
            return ("次の曲にした", True) if ok else ("いま音楽は鳴っていない", False)
        if name == "music_volume":
            return await self._volume(tool_input)
        return f"知らない道具：{name}", False

    async def _play(self, tool_input: dict) -> "tuple[str, bool]":
        said = str(tool_input.get("name") or "")
        row = music_rules.find_playlist(said, self._table())
        if row is None:
            return f"「{said}」は覚えていないので、かけられない", False
        title, uri, default_shuffle = row
        order = str(tool_input.get("order") or "")
        shuffle = music_rules.wants_shuffle(order or said, default_shuffle)
        # **鳴らす直前に機器をこちらへ**（MPRIS の口は現役になってから出る）。鍵の更新も
        # ここで起きる。切り替えられなくても鳴らしにいく（口が既に居れば鳴る）。
        if self._web is not None and self._device_name:
            with contextlib.suppress(Exception):
                self._web.activate(self._device_name)
        bus = self._bus()
        if not await self._io.play(bus, uri):
            return f"「{title}」をかけられなかった（音の出口が見つからない）", False
        await self._io.set_shuffle(bus, shuffle)
        self._mark(True)
        how = "ランダムで" if shuffle else ""
        logger.info("音楽：%s を%sかけ始めた", title, how or "順番に")
        return f"「{title}」を{how}かけ始めた", True

    def _mark(self, playing: bool) -> None:
        """鳴り始め・止まりを印す。**ここだけが溜める**（寿命と門が読む）。"""
        if self._state is None:
            return
        self._state.playing = playing
        if playing:
            self._state.started_at = self._now()

    async def _volume(self, tool_input: dict) -> "tuple[str, bool]":
        how = str(tool_input.get("how") or "")
        s = await self._io.status(self._bus())
        if not s:
            return "いま音楽は鳴っていない", False
        now = float(s.get("volume", 0.5))
        if "小さ" in how or "下げ" in how:
            value = max(0.0, now - VOLUME_STEP)
        else:
            value = min(1.0, now + VOLUME_STEP)
        await self._io.set_volume(self._bus(), value)
        return f"音量を {int(value * 100)}% にした", True
