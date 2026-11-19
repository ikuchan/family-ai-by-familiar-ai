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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..loop.paju_state import PajuState

logger = logging.getLogger(__name__)

# ログに載せる発話の長さ。記憶の内容と同じ扱いで、debug でも先頭だけにする。
_TRAIL_CHARS = 24
_SOUNDS = Path(__file__).resolve().parent.parent / "sounds"
_ALARM_WAV = _SOUNDS / "timer_alarm.wav"


async def _play_wav(path: Path, gain: float, *, quiet: bool = False) -> bool:
    """wav を 1 回鳴らす（差し替え点）。再生は `tools/tts` の sounddevice の口を借りる。

    `quiet` は短く繰り返す音（機械音 A・待ちの電子音）のため。鳴らした行を DEBUG にする（出-bu）。
    """
    from ..tools.tts import _play_via_sounddevice

    return await _play_via_sounddevice(str(path), gain, quiet=quiet)


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
        # 話しかけたときの合図（出-bg）。流している合図。返事の声はこれが終わってから流す。
        self._cue_task: asyncio.Task | None = None

    # ── 声 ────────────────────────────────────────────────────────────────

    @property
    def understands_tags(self) -> bool:
        """合成の担い手が角括弧タグを指示として解するか（`根拠台帳` §9）。

        整え方・`say` の説明・規則 `no-tts-tags` の3箇所がこの値を見る。声の担い手を
        知っているのは DIF なので、聞き先をここへ寄せる。
        """
        return bool(self._tts and self._tts.understands_tags)

    def speak_defs(self) -> list[dict]:
        """返事の道具（`say`）の定義。無い機体では空。"""
        return self._voice_defs("say")

    def _voice_defs(self, name: str) -> list[dict]:
        if not self._tts:
            return []
        return [d for d in self._tts.get_tool_definitions() if d.get("name") == name]

    async def speak(
        self, text: str, *, gain: float = 1.0, careful: bool = False, through_silence: bool = False
    ) -> "bool | None":
        """声に出す。返りは鳴ったか。黙っていて止めたら None（`_as_voice`）。

        **例外は飲む。** 機器は落ちる前提のもので、声が出せなかったことでターンごと
        壊すわけにはいかない（移す前と同じ扱い）。
        """
        if self._tts is None:
            return False
        tts = self._tts

        async def _say() -> bool:
            payload: dict = {"text": text}
            if gain != 1.0:
                payload["gain"] = gain  # タイマーの声だけ大きく（この 1 回の再生にだけ効く）
            if careful:
                # じっくり読む声（環-u）。主LLM が外への問い合わせの返りで話すときだけ。
                payload["careful"] = True
            result, _ = await tts.call("say", payload)
            # 合成器は成功なら `Said:` で始める。それ以外（API の 402・再生器が無い）は
            # 返り文字列にしか載らないので、ここで残さないと「声が出ない」が追えない。
            if not str(result).startswith("Said:"):
                logger.warning("DIF 声が出なかった：%s", str(result)[:160])
                return False
            return True

        async def _voice() -> bool:
            logger.debug("DIF speak → %s", text[:_TRAIL_CHARS])
            if gain != 1.0:
                logger.debug("DIF 声の倍率 %.2f（タイマーの知らせ・この 1 回だけ）", gain)
            started = time.monotonic()
            played = False
            try:
                played = bool(await self._ducked(_say))
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.warning("DIF 声で例外（声は出ずに続ける）", exc_info=True)
            # 合成＋再生の秒数は必ず残す（出-k-い）。返事が出るまでの体感にそのまま乗る。
            logger.info("DIF 声 %.2f 秒（%d 字）", time.monotonic() - started, len(text))
            return played

        try:
            return await self._as_voice(_voice, after_cue=True, through_silence=through_silence)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.warning("DIF 声の順番で例外（声は出ずに続ける）", exc_info=True)
            return False

    def _chart(self) -> "PajuState | None":
        """声の状態を持つ状態図（環-aw 第 5 段）。`ip` が無い、または状態図を持たない偽物なら None。"""
        from ..loop.paju_state import PajuState

        paju = getattr(self._ip, "_paju", None)
        sm = paju() if callable(paju) else None
        return sm if isinstance(sm, PajuState) else None

    async def _as_voice(
        self, play: "Callable[[], Awaitable[bool]]", *, after_cue: bool, through_silence: bool
    ) -> "bool | None":
        """声（答え、録音）を 1 本鳴らす（環-aw 第 5 段）。声の鍵を取ってから「話し始めた」を送り、鳴らし終えたら
        「話し終えた」を、鳴ったか（`played`）と一緒に送る。返りは鳴ったか。`after_cue` なら、鍵を取ったあと、
        合図が鳴り終えるのを待ってから話し始める（合図が鳴っているあいだは、声は合図のまま）。

        鍵を待っているあいだに打ち切られたら、どちらも送らない。鳴らしている途中で打ち切られても、「話し終えた」は
        打ち切りから守って送る（送らないと、声が話しているのまま残る）。状態図が無ければ、鍵も事象も無しに鳴らす。

        **黙っているあいだは、求めの外の声を鳴らさない。** 鍵を取ったあと、`through_silence`（開いている求めの声）が無く、
        黙っている依頼がいまの時刻で効いていれば、鳴らさず事象も送らずに None を返す（鳴らなかった＝偽と分ける。呼び手は
        None のとき、言ったことにしない）。黙るかは入口（`silence_hold`）でも見るので、ここで止めるのは閉じたあとの申し出と、
        自律機構のクレジットの知らせである。
        """
        sm = self._chart()
        if sm is None:
            if after_cue:
                await self._after_cue()
            return await play()
        if not through_silence:
            # 黙っている依頼の控えを戻し終えてから読む（戻す前に読むと、黙っていないと読む・第 2 段 項目 4）。
            # 始められなくても、鳴らすのはやめない（読めなければ黙っていない扱い・前のトニックと同じ）。
            with contextlib.suppress(Exception):
                await sm.start()
            # 鍵を待つ前にも見る。黙っていれば、ほかの声の終わりを待たずに返す（自律機構の輪は 0.5 秒ごとに知らせを
            # 試すので、待つと輪のタイマーが遅れる）。輪ごとに起きうるので、ログは開発者向けの段にする。
            if self._silenced(sm):
                logger.debug("DIF 黙っているので、求めの外の声を鳴らさない")
                return None
        async with sm.voice_turn():
            if not through_silence and self._silenced(sm):  # 鍵を待つあいだに黙った
                logger.debug("DIF 黙っているので、求めの外の声を鳴らさない")
                return None
            if after_cue:
                await self._after_cue()
            played = False
            try:
                await self._tell(sm, "voice_began")
                played = await play()
                return played
            finally:
                await asyncio.shield(self._tell(sm, "voice_ended", played=played))

    # ── 話しかけたときの合図（出-bg・2026-10-10 本人の決定）──────────────────────────────

    def start_thinking(self) -> "_Thinking":
        """Jev が判断しているあいだ、機械音 A を鳴らし続ける。返した印の `stop()` で止める。

        鳴らす仕事が、声の考え中の音の始まりと終わりを送る（環-aw 第 5 段）。
        """
        return _Thinking(asyncio.ensure_future(self._think_loop()))

    async def _think_loop(self) -> None:
        from ..core.reaction_cue import THINKING_WAV

        sm = self._chart()
        try:
            if sm is not None:
                await self._tell(sm, "thinking_cue_began")
            while True:
                # 音楽は絞らない（短い音を続けて鳴らすので、絞ると音量が上下し続ける）。
                if not await _play_wav(THINKING_WAV, 1.0, quiet=True):
                    return
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.debug("DIF 機械音が鳴らせなかった", exc_info=True)
        finally:
            if sm is not None:
                # 止めた直後に合図が始まっていれば、この終わりは移りが無く捨てられ、合図のまま残る
                await asyncio.shield(self._tell(sm, "thinking_cue_ended"))

    async def beep(self, kind: str) -> bool:
        """考えているの子 `kind` の電子音を 1 回（出-bu・環-aw 決定テ）。音楽は絞らない（短い音を続けるので）。

        その子に音が無ければ鳴らさず真を返す（鳴らせなかったのではない）。
        """
        from ..core.reaction_cue import BEEP_WAVS

        path = BEEP_WAVS.get(kind)
        if path is None:
            return True
        return await _play_wav(path, 1.0, quiet=True)

    def play_cue(self, cue: str) -> "asyncio.Future[None] | None":
        """決まった動作の合図を 1 回流す（待たない）。なしなら何もしない。音楽が鳴っていれば絞る。

        返りは鳴らしている仕事（鳴らさなければ None）。呼び手は鳴り終わった時刻を、つなぎの間に使う（出-bd ②）。
        """
        from ..core.reaction_cue import sound_for

        path = sound_for(cue)
        if path is None:
            return None

        sm = self._chart()

        async def _play() -> None:
            if sm is not None:
                await self._tell(sm, "cue_began")
            try:
                await _sound()
            finally:
                # 合図の終わりは、終わった仕事がいまの合図のときだけ送る。合図が 2 つ続いたとき、前の合図の終わりで
                # 後の合図を終わらせない（環-aw 第 5 段）。
                if sm is not None and self._cue_task is asyncio.current_task():
                    await asyncio.shield(self._tell(sm, "cue_ended"))

        async def _sound() -> None:
            async def _one() -> None:
                if not await _play_wav(path, 1.0):
                    logger.warning("DIF 合図が鳴らせなかった：%s（%s）", cue, path.name)

            try:
                if self._music_ducker is not None:
                    await self._music_ducker(_one)
                else:
                    await _one()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                # 鳴らせなかったことは回復した異常として残す（合図・黙る音が鳴らないまま気づけなくなる・出-dc）。
                logger.warning("DIF 合図が鳴らせなかった：%s", cue, exc_info=True)

        self._cue_task = asyncio.ensure_future(_play())
        logger.info("DIF 合図：%s（%s）", cue, path.name)
        return self._cue_task

    def set_music_ducker(self, ducker) -> None:
        """声のあいだ音楽を絞る口を挿す（知-aa）。挿さなければ、声はそのまま出る。"""
        self._music_ducker = ducker

    # ── タイマーの音 ──────────────────────────────────────────────────────

    @property
    def ringing(self) -> bool:
        return self._ring_task is not None and not self._ring_task.done()

    async def _after_cue(self) -> None:
        """合図（作り置きの声・効果音）が鳴っていれば、終わるのを待つ（出-bg・声を重ねない）。声と録音が使う。"""
        if self._cue_task is not None and not self._cue_task.done():
            with contextlib.suppress(Exception):
                await self._cue_task

    async def _ducked(self, play: "Callable[[], Awaitable[Any]]") -> Any:
        """音楽が鳴っていれば、`play` のあいだだけ絞る（知-aa・機械の反射）。声と録音が使う。"""
        if self._music_ducker is not None:
            return await self._music_ducker(play)
        return await play()

    @staticmethod
    async def _tell(sm: "PajuState", event: str, **data: Any) -> None:
        """声の事象を状態図へ送る（環-aw 第 5 段）。状態図が落ちても音は鳴らし、警告を 1 行残す（会話は止めない）。

        打ち切りはそのまま上へ返す。
        """
        try:
            await sm.fire(event, **data)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.warning("DIF 状態図へ %s を送れなかった（音は鳴らす）", event, exc_info=True)

    @staticmethod
    def _silenced(sm: "PajuState") -> bool:
        """いま黙っているか。状態図の依頼を、鳴らす時刻で決める（期限を過ぎた依頼では止めない）。"""
        from ..silence_state import is_silenced

        return is_silenced(sm.silence_request(), now=time.time())

    async def say_credit(self, name: str, *, gain: float) -> "bool | None":
        """残高切れを録音済みの声で知らせる（環-z・`sounds/credit_<担い手>.wav`）。LLM も TTS も通さない。

        録音が無ければ鳴らさず偽を返す（落とさない）。録音は `scripts/gen_credit_voices.py` が作る。
        求めの外の声なので、黙っているあいだは鳴らさず None を返す（`_as_voice`）。
        """
        # クレジット切れの知らせは、いままでどおり合図を待たず、音楽も絞らない。
        played = await self.say_recorded(f"credit_{name}", gain=gain, like_speech=False)
        if played is not None:  # 黙っていて止めたときは、輪ごとに試すので残さない
            logger.info("DIF クレジット切れを知らせた：%s", name)
        return played

    async def say_recorded(
        self, name: str, *, gain: float, like_speech: bool = True, through_silence: bool = False
    ) -> "bool | None":
        """録音済みの声 `sounds/<name>.wav` を鳴らす。LLM も TTS も通さない（クレジット切れ・環-z、時間切れの一言・環-aw
        第 1 段 項目 8）。いまの声の録音があればそれを使う（出-bv）。

        `like_speech`＝声（`speak`）と同じく、合図が終わるのを待ち、音楽を絞って鳴らす。
        録音が無い・鳴らせなければ偽を返す（落とさない）。黙っていて止めたら None（`_as_voice`）。
        """
        from ..core.reaction_cue import voiced

        path = voiced(_SOUNDS / f"{name}.wav")  # いまの声の録音（出-bv）。無ければ SBV2 版
        if not path.exists():
            logger.warning("DIF 録音が無い：%s", path.name)
            return False

        async def _voice() -> bool:
            try:
                if like_speech:
                    return bool(await self._ducked(lambda: _play_wav(path, gain)))
                return await _play_wav(path, gain)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.warning("DIF 録音を鳴らせなかった：%s", path.name, exc_info=True)
                return False

        return await self._as_voice(_voice, after_cue=like_speech, through_silence=through_silence)

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

    def device(self, kind: str, content: str, *, passes_gate: bool = False) -> None:
        """機器が出した出来事（タイマー、アラーム、音楽の知らせ）を I の待ち行列へ入れる。

        外から I へ入る出来事なので DIF の担当である（T が渡すのは、見張るのが T の
        常駐の仕事だからにすぎない）。
        """
        logger.debug("DIF device → %s／%s", kind, content[:_TRAIL_CHARS])
        self._ip.push_device(kind, content, passes_gate=passes_gate)

    def record(self, kind: str, content: str) -> None:
        """機器の出来事を、求めを立てずに記憶へ記録だけする（出-as §2.7・人の出入り）。"""
        logger.debug("DIF record → %s／%s", kind, content[:_TRAIL_CHARS])
        self._ip.note_device(kind, content)


__all__ = ["DIF"]


class _Thinking:
    """鳴らし続けている機械音 A の印（出-bg）。`stop()` は何度呼んでもよい。"""

    def __init__(self, task: "asyncio.Future[None]") -> None:
        self._task = task

    def stop(self) -> None:
        if not self._task.done():
            self._task.cancel()
