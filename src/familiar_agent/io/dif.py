"""外部機器接続 DIF：I と外の機械をつなぐ唯一の口（`設計図` ③-2）。

設計は口を4つ（IIF・DIF・AIF・OIF）と定め、この4つ以外にコンポーネントどうしが直接
つながる線を置かない。ところが**外部の機械へは口が無く**、`loop/event_loop.py` が
スピーカーと調べものの道具を直接掴んでいた。

**挙動は変えない。** 返り値の形も、例外の畳み方も、順序もそのままで、口は転送し、
通ったものを debug に残すだけである（AIF と同じ作り）。

**主LLM は入らない**（出-c でモデルは資源とした）。記憶は OIF の担当である。

段1で通すのは**声と調べもの**だけである。知覚（`see`・`look`）は 知-c が実機で挙動を
追っている最中なので、そこが済むまで触らない——同じ箇所を切り出しと機能変更の両方で
触ると、どちらで挙動が変わったのかを切り分けられなくなる。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable  # noqa: TC003  型注記に使う
from typing import Any

import asyncio
import contextlib
import re
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ログに載せる発話の長さ。記憶の内容と同じ扱いで、debug でも先頭だけにする。
_TRAIL_CHARS = 24
_ALARM_WAV = Path(__file__).resolve().parent.parent / "sounds" / "timer_alarm.wav"


async def _play_wav(path: Path, gain: float) -> bool:
    """wav を 1 回鳴らす（差し替え点）。再生は `tools/tts` の sounddevice の口を借りる。"""
    from ..tools.tts import _play_via_sounddevice

    return await _play_via_sounddevice(str(path), gain)


_SESSION_MARK = re.compile(r"\s*──（\d{4}-\d{2}-\d{2} のセッション／[^）]*）\s*$")


def strip_session_mark(text: str) -> str:
    """`ask_vault_*` の返りの末尾の素性の印を落とす（読み上げない・`家の記録との接続` §2）。"""
    return _SESSION_MARK.sub("", text or "").rstrip()


class DIF:
    """I と外部の機械の唯一の出入り口。

    **要るものだけを受け取る。** `agent` を丸ごと持てば、口はその 89 個の属性すべてに
    手が届く。届く必要のないものへ届く形は、口を1枚挟んだ意味を消す。

    `tts` は声の担い手（無い機体では `None`）、`search` と `fetch` は調べものの道具、
    `mcp` は MCP の道具を持つ側。どれも `agent` の `__init__` で一度作られたきりで
    差し替わらないので、写しを持つ。`ip` は機器の出来事を受ける側の I（`InformationProcessing`）で、
    外から中へ入る向きだけが使う。

    **持つ側ごとに1つ作る。** I は外へ出る向き（`tts`・`search`・`fetch`・`mcp`）を、
    T は中へ入る向き（`ip`）を持つ。AIF が `tonic.py` と `agent.py` に1つずつ在るのと
    同じで、口は薄い転がしなので、要る向きだけを持てばよい。
    """

    def __init__(self, *, tts=None, search=None, fetch=None, mcp=None, ip=None) -> None:
        self._tts = tts
        self._search = search
        self._fetch = fetch
        self._mcp = mcp
        self._ip = ip
        self._ring_task: asyncio.Task | None = None  # 鳴っているタイマーの音（知-n-ろ）
        #: 声を出すあいだ音楽を絞る口（知-aa・2026-09-21）。無ければ何もしない。
        self._music_ducker: "Callable[[Callable[[], Awaitable[Any]]], Awaitable[Any]] | None" = None
        # 山谷の既定（小さくする秒・戻す秒・その倍率）。T が Config から入れる
        self._ring_shape = (8.0, 25.0, 0.1)

    # ── 声 ────────────────────────────────────────────────────────────────

    @property
    def understands_tags(self) -> bool:
        """合成の担い手が角括弧タグを指示として解するか（`根拠台帳` §9）。

        整え方・`say` の説明・規則 `no-tts-tags` の3箇所がこの値を見る。声の担い手を
        知っているのは DIF なので、聞き先をここへ寄せる。
        """
        return bool(self._tts and self._tts.understands_tags)

    def speak_defs(self) -> list[dict]:
        """声の道具の定義。無い機体では空。"""
        return self._tts.get_tool_definitions() if self._tts else []

    async def speak(self, text: str, *, gain: float = 1.0, careful: bool = False) -> None:
        """声に出す。

        **例外は飲む。** 機器は落ちる前提のもので、声が出せなかったことでターンごと
        壊すわけにはいかない（移す前と同じ扱い）。
        """
        if self._tts is None:
            return
        logger.debug("DIF speak → %s", text[:_TRAIL_CHARS])
        if gain != 1.0:
            logger.debug("DIF 声の倍率 %.2f（タイマーの知らせ・この 1 回だけ）", gain)
        started = time.monotonic()

        async def _say() -> None:
            payload: dict = {"text": text}
            if gain != 1.0:
                payload["gain"] = gain  # タイマーの声だけ大きく（この 1 回の再生にだけ効く）
            if careful:
                # じっくり読む声（環-u）。主LLM が外への問い合わせの返りで話すときだけ。
                payload["careful"] = True
            result, _ = await self._tts.call("say", payload)
            # 合成器は成功なら `Said:` で始める。それ以外（API の 402・再生器が無い）は
            # 返り文字列にしか載らないので、ここで残さないと「声が出ない」が追えない。
            if not str(result).startswith("Said:"):
                logger.warning("DIF 声が出なかった：%s", str(result)[:160])

        with contextlib.suppress(Exception):
            # 音楽が鳴っていれば、声のあいだだけ絞る（知-aa・機械の反射）。
            if self._music_ducker is not None:
                await self._music_ducker(_say)
            else:
                await _say()
        # 合成＋再生の秒数は必ず残す（出-k-い）。返事が出るまでの体感にそのまま乗る。
        logger.info("DIF 声 %.2f 秒（%d 字）", time.monotonic() - started, len(text))

    def set_music_ducker(self, ducker) -> None:
        """声のあいだ音楽を絞る口を挿す（知-aa）。挿さなければ、声はそのまま出る。"""
        self._music_ducker = ducker

    # ── タイマーの音 ──────────────────────────────────────────────────────

    @property
    def ringing(self) -> bool:
        return self._ring_task is not None and not self._ring_task.done()

    def configure_ring(self, *, soft_after: float, soft_until: float, soft_gain: float) -> None:
        """山谷の形を Config から入れる（`RING_SOFT_AFTER_SEC`・`RING_SOFT_UNTIL_SEC`・`RING_SOFT_GAIN`）。"""
        self._ring_shape = (float(soft_after), float(soft_until), float(soft_gain))

    def ring(
        self,
        *,
        seconds: float,
        gain: float = 1.0,
        soft_after: "float | None" = None,
        soft_until: "float | None" = None,
        soft_gain: "float | None" = None,
    ) -> None:
        """タイマーの音（`sounds/timer_alarm.wav`・1 秒）を `seconds` のあいだ繰り返す（知-n-ろ）。

        山谷（2026-09-19）：`soft_after` 秒で `soft_gain` 倍に小さくし、`soft_until` 秒を過ぎたら元の
        大きさに戻す（`core/ring_rules.gain_at`）。繰り返しごとにその時点の倍率で再生する。

        「タイマーです」の一言の代わり。**声の口（`speak`）は通らない**——通すとマイクの門
        （`tts_active`）が立ち、鳴っている最中の「止めて」が届かない。音をマイクが拾う分は
        STT の幻聴の門（`no_speech_prob`）で落ちる想定（実機で確かめる）。止めるのは
        `stop_ring`（`cancel_timer`・`/timer stop`）か時間切れ。
        """
        self.stop_ring()
        if seconds <= 0:
            return
        shape = self._ring_shape
        soft_after = shape[0] if soft_after is None else soft_after
        soft_until = shape[1] if soft_until is None else soft_until
        soft_gain = shape[2] if soft_gain is None else soft_gain
        logger.info(
            "DIF タイマーの音を鳴らす（%.0f 秒・倍率 %.2f・%.0f〜%.0f 秒は %.2f）",
            seconds,
            gain,
            soft_after,
            soft_until,
            soft_gain,
        )
        self._ring_task = asyncio.create_task(
            self._ring(seconds, gain, soft_after, soft_until, soft_gain)
        )

    def stop_ring(self) -> None:
        if self.ringing:
            logger.info("DIF タイマーの音を止めた")
            self._ring_task.cancel()  # type: ignore[union-attr]
        self._ring_task = None

    async def _ring(
        self, seconds: float, gain: float, soft_after: float, soft_until: float, soft_gain: float
    ) -> None:
        from ..core.ring_rules import gain_at

        started = time.monotonic()
        deadline = started + seconds
        try:
            while time.monotonic() < deadline:
                now_gain = gain_at(
                    time.monotonic() - started,
                    gain,
                    soft_after=soft_after,
                    soft_until=soft_until,
                    soft_gain=soft_gain,
                )
                ok = await _play_wav(_ALARM_WAV, now_gain)
                if not ok:
                    logger.warning("DIF タイマーの音が出なかった：%s", _ALARM_WAV)
                    return
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("DIF タイマーの音で例外：%s", e)

    # ── 調べもの ──────────────────────────────────────────────────────────

    def lookup_defs(self, kind: str) -> list[dict]:
        """調べものの道具の定義。"""
        tool = self._search if kind == "search_deferred" else self._fetch
        return tool.get_tool_definitions()

    async def lookup(self, kind: str, params: dict) -> tuple[str, bool]:
        """調べものを投げる。返りは (文面, 背景タスクを作ったか)。

        **ここでは畳まない。** 投げられなかったときに開いた意図を閉じるのは呼び手の
        仕事で、口が例外を飲むと飛行中の数が合わなくなる。
        """
        tool = self._search if kind == "search_deferred" else self._fetch
        logger.debug("DIF lookup → %s", kind)
        return await tool.dispatch(params)

    # ── MCP の道具 ────────────────────────────────────────────────────────

    async def call_tool(self, name: str, params: dict) -> tuple[str, bool]:
        """MCP の同期の道具を呼び、(文面, 使えたか) を返す（`get_house_rules`・`get_family_schedule`）。

        口が無ければ (「繋がっていない」, False)。MCP のサーバーは落ちる前提のもので、
        `MCPClientManager.call_result` は例外を投げずに失敗を印（`ok`）で返す（出-o）。
        失敗の本文は生のエラー文で、**人へ渡す情報ではない**（対処できない）。呼び手は
        ログに残し、「道具が使えなかった」として扱う。
        """
        if self._mcp is None:
            return f"（{name} は繋がっていない）", False
        logger.debug("DIF call_tool → %s", name)
        r = await self._mcp.call_result(name, dict(params or {}))
        # 記録の道具（`ask_vault_*`）は末尾に素性の印 `──（YYYY-MM-DD のセッション／継続）` を付ける。
        # 本人向けの情報ではないので、W へ載せる前に落とす（知-g-い）。
        return strip_session_mark(str(r.text)), bool(r.ok)

    def tool_defs_with_prefix(self, prefix: str) -> list[dict]:
        """名前が `prefix` で始まる MCP の道具を全部取り出す（`ask_vault_`・話者ゲートは呼び手が掛ける）。"""
        if self._mcp is None:
            return []
        with contextlib.suppress(Exception):
            return [
                d
                for d in self._mcp.get_tool_definitions()
                if str(d.get("name", "")).startswith(prefix)
            ]
        return []

    def tool_defs(self, name: str) -> list[dict]:
        """MCP の道具を**名前で1本だけ**取り出す。

        **話者ゲートはこちら側の責任である。** サーバー側からは誰が話しているか見えない
        ので、個人ティアの道具名には人が入っている（`ask_vault_yusuke`）。**名前に人が
        入っている道具は、その人のターン以外では出さない**——`description` でお願いする
        のではなく、定義リストから落とす。存在しない道具は呼べない。

        MCP のサーバーは落ちる前提のもので、引けなければ空で返す。
        """
        if self._mcp is None:
            return []
        with contextlib.suppress(Exception):
            return [d for d in self._mcp.get_tool_definitions() if d.get("name") == name]
        return []

    # ── 機器の出来事（外 → I） ─────────────────────────────────────────────

    def device(
        self, kind: str, content: str, *, release_pending: bool = False, passes_gate: bool = False
    ) -> None:
        """人の出入りなど、機器が出した出来事を I の待ち行列へ入れる。

        カメラが出す事実なので DIF の担当である（T が渡すのは、時計を持つのが T
        だからにすぎない）。
        """
        logger.debug("DIF device → %s／%s", kind, content[:_TRAIL_CHARS])
        self._ip.push_device(
            kind, content, release_pending=release_pending, passes_gate=passes_gate
        )


__all__ = ["DIF"]
