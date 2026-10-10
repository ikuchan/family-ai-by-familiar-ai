"""音楽の道具（知-aa 段 1・2026-09-21・`ユースケース④_音楽再生`）。

主LLM が呼ぶ 4 本——`play_music`（かける）・`stop_music`（止める）・`next_track`（次へ）・
`music_volume`（音量）。鳴らす先は `spotifyd`（MPRIS・`io/music`）で、**状態は溜めず、必要な
ときに読む**（鳴っているか・いまの曲・音量）。持ち続けるのは音楽の意図だけである。

プレイリストは人が書く `MUSIC.md`（`名前：URI[：ランダム]`）から当てる。順番かランダムかは、
**表の 3 つ目の欄が既定で、言われた言葉がそのつど上書きする**（本人の決定・2026-09-21）。
"""

from __future__ import annotations

import asyncio
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
            "音楽をかける（「ケイマンかけて」「米津玄師かけて」「Lemon かけて」）。"
            "name は言われた名前（プレイリスト・アーティスト・アルバム・曲）。覚えているプレイリスト、"
            "あなたのプレイリストとライブラリ、プレイリストに入っているアーティストの順に探し、無ければ "
            "Spotify 全体から探す。kind に「曲」「アーティスト」「アルバム」「プレイリスト」を渡すと、全体から"
            "探すときの種類になる。order に「ランダム」か「順番」を渡すと、そのときだけ順番を変えられる。"
            "返りで何をどこから見つけたかを言うので、違えば言い直してもらう。"
            "名前を言われなければ（「音楽かけて」「何か曲かけて」）name は空にする——Spotify で止まっていた続きをかける。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "言われた名前"},
                "kind": {
                    "type": "string",
                    "description": "「曲」「アーティスト」「アルバム」「プレイリスト」（省略可・省けば曲）",
                },
                "order": {"type": "string", "description": "「ランダム」か「順番」（省略可）"},
            },
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
    {
        "name": "music_suggestion_reply",
        "description": (
            "すすめた曲への返事を受ける（知-aa）。「いいね」「かけて」なら reply に「気に入った」——その曲をかけて、"
            "気に入った曲として覚える。「いらない」「いまはいい」なら「いらない」——その曲は二度とすすめない。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reply": {"type": "string", "description": "「気に入った」か「いらない」"}
            },
            "required": ["reply"],
        },
    },
]


#: 鳴っていないときの `[音楽]` の枠（知-ak 段 5）。
NOT_PLAYING = "[音楽] いまは何も鳴っていない"


#: 鳴らし始めたあと、頼んだものが鳴っているかを見る長さと間隔（秒・知-al）。〔仮・本人の決定〕
_CONFIRM_SEC = 3.0
_CONFIRM_TICK = 0.5


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
        spotifyd_conf: str = "",
        spotifyd_wait_sec: float = 10.0,
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
        # 鳴らす前に spotifyd が居るかを見て、居なければ立ち上げる（2026-10-07）。空なら見ない（試験の土台）。
        self._spotifyd_conf = spotifyd_conf
        self._spotifyd_wait = spotifyd_wait_sec

    def get_tool_definitions(self) -> list[dict]:
        return [dict(d) for d in TOOL_DEFINITIONS]

    async def frame(self, status: "dict | None" = None) -> str:
        """`[音楽]` の枠。読んだ様子（`status`）があればそれを使う（読み直さない）。読めなければ空。

        **鳴っていないことも書く**（知-ak 段 5・2026-10-07 実機 18:43）。書かないと、記憶の「かけ始めた」がいまの
        状態のように読まれ、`play_music` を呼ばずに「もうかけてますよ」と答えた。
        """
        s = status if status is not None else await self._io.status(self._bus())
        if not s:
            return ""
        if not s.get("playing"):
            return NOT_PLAYING
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
        if name == "music_suggestion_reply":
            return await self._reply(str(tool_input.get("reply") or ""))
        return f"知らない道具：{name}", False

    async def _play(self, tool_input: dict) -> "tuple[str, bool]":
        """近いところから順に探してかける（知-aa 段 3・本人の決定）。

        1 `MUSIC.md`（表記ゆれを均す）→ 2 自分のプレイリスト → 3 ライブラリ → 4 プレイリストに入っている曲と
        アーティスト → 5 Spotify 全体の検索（プレイリストとライブラリに居るアーティストを先に選ぶ）。
        正解はあなたが作ったプレイリストや、そこに入っている人であることが多い（本人）。返りで、何をどこから
        見つけたかを言う。
        """
        from ..core import music_catalog

        said = str(tool_input.get("name") or "").strip()
        if not said or music_rules.means_no_name(said):  # 「音楽」「曲」は名前ではない（知-al (2)）
            return await self._resume()
        order = str(tool_input.get("order") or "")
        catalog = music_catalog.stored()
        local = music_rules.find_local(said, self._table(), catalog)
        if local is not None:
            source, label, uri, default_shuffle = local
            shuffle = music_rules.wants_shuffle(order or said, default_shuffle)
            heading = {
                "MUSIC.md": f"「{label}」を",
                "プレイリスト": f"あなたのプレイリスト「{label}」を",
                "ライブラリ": f"ライブラリの{label}を",
                "プレイリストの曲": f"プレイリストに入っている{label}を",
                # 読みで当てたときは、そう言う（知-at）。違えば言い直してもらえるように。
                "読み:MUSIC.md": f"「{said}」を、読みから「{label}」と取って",
                "読み:プレイリスト": f"「{said}」を、読みからあなたのプレイリスト「{label}」と取って",
            }[source]
        else:
            found = await self._search(said, str(tool_input.get("kind") or ""), catalog)
            if found is None:
                return f"「{said}」は見つからなかったので、かけられない", False
            heading, uri = found
            shuffle = music_rules.wants_shuffle(order, False)
        failed = await self._start(heading, uri)
        if failed is not None:
            return failed
        bus = self._bus()
        if ":playlist:" in uri or ":album:" in uri or ":artist:" in uri:
            await self._io.set_shuffle(bus, shuffle)
        self._mark(True)
        how = "ランダムで" if shuffle else ""
        logger.info("音楽：%s%sかけ始めた（%s）", heading, how or "", uri)
        return f"{heading}{how}かけ始めた", True

    async def _start(self, heading: str, uri: str) -> "tuple[str, bool] | None":
        """`uri` を鳴らし始める。鳴らせたら None、できなければ返りの文（失敗）。`_play` とすすめた曲の返事が使う。

        **Web API で始め、鳴ったものを確かめる**（知-al・2026-10-07）。MPRIS の `OpenUri` では曲の URI に切り替わらず、
        前のアルバムが鳴ったのに「かけましたよ」と答えた（実機 22:01）。Web API が無い・失敗したら MPRIS の道へ。
        """
        await self._ensure_spotifyd()
        started = False
        if self._web is not None and self._device_name:
            try:
                started = bool(await asyncio.to_thread(self._web.play, self._device_name, uri))
            except Exception:  # noqa: BLE001
                logger.warning(
                    "音楽：Web API で鳴らし始められなかった（MPRIS で鳴らす）", exc_info=True
                )
        if started:
            if not await self._confirm(uri):
                logger.info("音楽：%sかけようとしたが、別の曲が鳴っている（%s）", heading, uri)
                return f"{heading}かけようとしたが、別の曲が鳴っている", False
            return None
        # **鳴らす直前に機器をこちらへ**（MPRIS の口は現役になってから出る）。鍵の更新もここで起きる。
        if self._web is not None and self._device_name:
            with contextlib.suppress(Exception):
                self._web.activate(self._device_name)
        if not await self._io.play(self._bus(), uri):
            return f"{heading}かけられなかった（音の出口が見つからない）", False
        return None

    async def _resume(self) -> "tuple[str, bool]":
        """曲名なし：Spotify で止まっていた続きを鳴らす（出-ay 段 4-4e・2026-10-09）。

        Web API で曲を指定せずに再生を頼み、鳴り始めたかを確かめる。Web API は失敗しても空を返すので、頼めたかでは
        なく鳴っているかで見る。続きが無い・Web API が無いときは、かけられなかったと返す（本人の決定ア）。
        """
        if self._web is None or not self._device_name:
            return "続きをかけられなかった（Spotify につながっていない）", False
        await self._ensure_spotifyd()
        try:
            asked = bool(await asyncio.to_thread(self._web.resume, self._device_name))
        except Exception:  # noqa: BLE001
            logger.warning("音楽：続きを頼めなかった", exc_info=True)
            asked = False
        if not asked or not await self._confirm(None):
            logger.info("音楽：止まっていた続きが鳴らなかった")
            return "止まっていた続きが無く、かけるものが無かった", False
        self._mark(True)
        logger.info("音楽：止まっていた続きをかけ始めた")
        return "止まっていた続きをかけ始めた", True

    async def _confirm(self, uri: "str | None") -> bool:
        """Spotify 側でいま鳴っているもの（曲かその上）が `uri` か（None なら鳴っているか）。`_CONFIRM_TICK` おきに最大 `_CONFIRM_SEC` 見る。"""
        waited = 0.0
        while True:
            with contextlib.suppress(Exception):
                now = await asyncio.to_thread(self._web.now_playing)
                if uri is None and now.get("playing"):
                    return True  # 続き：何が鳴るかは決めていないので、鳴っているかだけを見る
                if uri is not None and uri in (now.get("item"), now.get("context")):
                    return True
            if waited >= _CONFIRM_SEC:
                return False
            await asyncio.sleep(_CONFIRM_TICK)
            waited += _CONFIRM_TICK

    async def _ensure_spotifyd(self) -> None:
        """spotifyd が居なければ立ち上げ、機器が Spotify に見えるまで待つ（2026-10-07 実機・最大 10 秒〔仮〕）。"""
        if not self._spotifyd_conf:
            return
        from ..io import spotifyd

        state = await asyncio.to_thread(spotifyd.ensure_running, self._spotifyd_conf)
        if state == "started" and self._web is not None and self._device_name:
            await spotifyd.wait_for_device(
                self._web, self._device_name, wait_sec=self._spotifyd_wait
            )

    async def _search(self, said: str, kind_word: str, catalog) -> "tuple[str, str] | None":
        """手元に無いとき：プレイリストに入っているアーティストなら、その人を。無ければ全体から探す。"""
        if self._web is None:
            return None
        artist = music_rules.playlist_artist_in(said, catalog)
        if artist is not None:
            hits = await asyncio.to_thread(self._web.search, artist, "artist")
            pick = next((h for h in hits if music_rules.same_name(h["title"], artist)), None)
            if pick is not None:
                return f"プレイリストに入っている{artist}を", str(pick["uri"])
        kind = music_rules.KINDS.get(kind_word.strip(), "track")
        hits = await asyncio.to_thread(self._web.search, said, kind)
        if not hits:
            return None
        mine = music_rules.my_artists(catalog)
        pick = next((h for h in hits if music_rules._norm(h.get("artist", "")) in mine), hits[0])
        label = f"「{pick['title']}」" + (f"（{pick['artist']}）" if pick.get("artist") else "")
        return f"ライブラリに無かったので、Spotify から探した{label}を", str(pick["uri"])

    async def _reply(self, reply: str) -> "tuple[str, bool]":
        """すすめた曲への返事（知-aa 段 4）。気に入ったらかけて控え、いらなければ二度とすすめない。"""
        from ..core import music_suggestion as ms

        s = ms.stored()
        c = s.candidate
        if c is None:
            return "いま勧めている曲は無い", False
        song = {"title": c.title, "artist": c.artist, "uri": c.uri}
        s.candidate = None
        if "いらな" in reply or "いい" == reply.strip() or "結構" in reply:
            s.declined.append(song)
            ms.store(s)
            return f"「{c.title}」はもう勧めない", True
        s.liked.append(song)
        ms.store(s)
        # play_music と同じ道で鳴らし、鳴ったかを確かめる（出-ay 段 4-4f・知-al の穴を塞ぐ）。返りに曲名を入れる。
        failed = await self._start(f"すすめた「{c.title}」（{c.artist}）を", c.uri)
        if failed is not None:
            return failed
        self._mark(True)
        return f"すすめた「{c.title}」（{c.artist}）をかけ始めた", True

    def _mark(self, playing: bool) -> None:
        """鳴り始め・止まりを印す。**ここだけが溜める**（寿命と門が読む）。"""
        if self._state is None:
            return
        self._state.playing = playing
        if playing:
            self._state.started_at = self._now()
            self._state.last_title = ""  # かけ直しでも最初の曲を書く（知-aa 段 2）

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
