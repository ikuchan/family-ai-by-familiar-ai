"""音楽の見張りと減音（知-aa 段 1・2026-09-21）。

**寿命**：鳴らし始めてから 30 分で止め、止めたことを一言言う（本人の決定）。T の刻みから
呼ぶ（ストップウォッチと同じ形・`loop/stopwatch_watch.py`）。

**減音**：パジュが声を出しているあいだ（と、その後 10 秒）は音量を 2 分の 1 に絞り、基準へ戻す。
減音の基準はそのつど読む（人が動かした値を踏み潰さない）。ふだんの音量は、人が声で変えたときと止めたときに
記録し、かけ始めに使う（`core/music_state`・知-bi）。これは機械の反射で、主LLM は通らない（`ユースケース④` [D-会話減音]）。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ..core import music_state
from ..core.music_rules import RESTORE_AFTER_SEC, duck, expired

logger = logging.getLogger(__name__)


async def check_music_expired(
    *, io: Any, bus: Any, state: Any, now: float, say: Callable[[str], Any]
) -> bool:
    """30 分を過ぎていたら止めて一言言う。止めたら True。"""
    if not music_state.is_playing(state):
        return False
    started = float(getattr(state, "started_at", 0.0) or 0.0)
    if not started or not expired(started, now=now):
        return False
    s: "dict | None" = None
    with contextlib.suppress(Exception):  # 止める前に読む（止めたときの音量を覚える・知-bi）
        s = await io.status(bus)
    await io.stop(bus)
    state.playing = False
    base = music_state.base_volume(state, s) if s else None
    if base is not None:
        music_state.save_volume(base)
    logger.info("音楽：30 分たったので止めた")
    # 言い方は軽量LLM が書くので、事実の形で渡す（知-bh・2026-10-16）。
    say("音楽：30 分たったので止めた")
    return True


async def duck_while_speaking(
    *,
    io: Any,
    bus: Any,
    state: Any,
    speak: Callable[[], Awaitable[Any]],
    sleep: "Callable[[float], Awaitable[None]] | None" = None,
    busy: "Callable[[], bool] | None" = None,
    why: str = "話す",
) -> Any:
    """声と知らせの音（環-aw 第 6 段 項目 2）を出すあいだだけ音楽を絞る。鳴っていなければ何もしない。

    絞る前に**そのときの音量を読む**（人が変えていれば、その値へ戻す）。終わってから `RESTORE_AFTER_SEC` 秒おいて
    戻す——すぐ戻すと、続けて話すたびに上下してうるさい。

    **戻しは裏で予約し、待たずに返す**（出-bd ①）。予約中に次の声が来たら予約を取り消し、絞ったまま基準
    （`ducked_base`）を引き継ぐ。声と知らせの音は時間が重なりうるので、絞っている最中（`ducked_base` がある）なら
    2 度は絞らない。戻す仕事は、戻す前に `busy`（知らせの音が鳴っているか、声が話しているか。持ち主を読む）を見て、
    忙しければ待ち直す（決定ヌの禁止の条件）。戻す値は、そのときの `ducked_base`（絞っているあいだに人が変えた値）。
    """
    # **絞れなくても声は出す。** ここで例外を外へ出すと、`DIF.speak` の例外抑止に飲まれて
    # 声そのものが消える（テストが 4 件捕まえた・2026-09-21）。音楽は添え物で、声が本体である。
    try:
        with contextlib.suppress(Exception):
            pending = getattr(state, "restore_task", None)
            if isinstance(pending, asyncio.Task) and not pending.done():
                pending.cancel()  # 絞ったまま引き継ぐ
            if getattr(state, "ducked_base", None) is None and music_state.is_playing(state):
                s = await io.status(bus)
                # 読んでいるあいだに、ほかの者（声か知らせの音）が絞っていたら、2 度は絞らない
                if s and s.get("playing") and state.ducked_base is None:
                    base = float(s.get("volume", 0.5))
                    state.ducked_base = base  # 絞る前に置く（絞る途中で打ち切られても戻せる）
                    try:
                        await io.set_volume(bus, duck(base))
                    except Exception:
                        state.ducked_base = None  # 絞れなかった。絞っている最中に見せない
                        raise
                    # 効いたかを後から確かめられるように残す（知-ak-ろ・2026-10-07）
                    logger.info("音楽：%sあいだ音量を下げた（%.2f → %.2f）", why, base, duck(base))
        return await speak()
    finally:
        if getattr(state, "ducked_base", None) is not None:
            with contextlib.suppress(Exception):
                pending = getattr(state, "restore_task", None)
                if isinstance(pending, asyncio.Task) and not pending.done():
                    pending.cancel()
                state.restore_task = asyncio.ensure_future(
                    _restore_later(io=io, bus=bus, state=state, sleep=sleep, busy=busy)
                )


async def _restore_later(
    *,
    io: Any,
    bus: Any,
    state: Any,
    sleep: "Callable[[float], Awaitable[None]] | None",
    busy: "Callable[[], bool] | None" = None,
) -> None:
    """`RESTORE_AFTER_SEC` 秒おいて基準へ戻す（裏の仕事）。まだ鳴っているか話していれば待ち直す（環-aw 第 6 段）。

    取り消されたら戻さず、基準も消さない（次の者が絞ったまま使う）。
    """
    wait = sleep or asyncio.sleep
    await wait(RESTORE_AFTER_SEC)
    for _ in range(_MAX_REWAITS):
        if busy is None or not _busy(busy):
            break
        await wait(RESTORE_AFTER_SEC)
    else:
        # 声の終わりの事象が落ちて話しているのまま残るなどで、いつまでも忙しく見えるとき。絞ったまま残すより戻す。
        logger.warning("音楽：%d 回待っても静かにならないので音量を戻す", _MAX_REWAITS)
    base = getattr(state, "ducked_base", None)
    if base is None:
        return
    try:
        await io.set_volume(bus, base)
        logger.info("音楽：音量を戻した（%.2f）", base)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        logger.warning("音楽の音量を戻せなかった", exc_info=True)
    state.ducked_base = None


#: 戻す仕事が待ち直す上限（`RESTORE_AFTER_SEC` ごと。10 秒 × 30 回＝5 分。知らせの音は 30 秒）。
_MAX_REWAITS = 30


def _busy(busy: "Callable[[], bool]") -> bool:
    """まだ鳴っているか話しているか。読めなければ忙しくないとみなす（戻さずに残るより、戻すほうがよい）。"""
    try:
        return bool(busy())
    except Exception:  # noqa: BLE001
        return False


async def observe(*, io: Any, bus: Any, state: Any, record: Callable[[str], Any]) -> dict:
    """鳴っているあいだの様子を読み、曲送りと鳴り終わりを記録する（知-aa 段 2）。いまの様子を返す。

    MPRIS を読むのはここだけで、T の見張り（30 秒ごと）と反復が呼ぶ。黙って聴いているあいだの曲も
    残す（本人の決定イ）。曲は最後に書いた曲名と違うときだけ書く。止まっていたら印を下ろし（声の入口の
    門が開く）、声には出さずに記録する。**読めないときは何もしない**——機器は落ちる前提で、読めない
    ことを「止まった」とは見ない。鳴らしていなければ読まない。
    """
    if not music_state.is_playing(state):
        return {}
    try:
        s = await io.status(bus)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        logger.warning("音楽：いまの様子を読めなかった", exc_info=True)
        return {}
    if not s or not s.get("playing"):
        # 鳴り止んだときの音量を覚える（知-bi）。読めなければ覚えない（スマホから止めて spotifyd が居ないなど）。
        base = music_state.base_volume(state, s) if s else None
        if base is not None:
            music_state.save_volume(base)
        state.playing = False
        logger.info("音楽：止まっていたので鳴っている印を下ろした")
        record("音楽が止まった")
        return s or {}
    title = str(s.get("title") or "")
    if title and title != getattr(state, "last_title", ""):
        state.last_title = title
        artist = str(s.get("artist") or "")
        record(f"音楽：{title}／{artist}" if artist else f"音楽：{title}")
    return s
