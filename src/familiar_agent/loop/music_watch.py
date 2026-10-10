"""音楽の見張りと減音（知-aa 段 1・2026-09-21）。

**寿命**：鳴らし始めてから 30 分で止め、止めたことを一言言う（本人の決定）。T の刻みから
呼ぶ（ストップウォッチと同じ形・`loop/stopwatch_watch.py`）。

**減音**：パジュが声を出しているあいだ（と、その後 10 秒）は音量を 4 分の 1 に絞り、基準へ戻す。
基準は**人が変えた値**をそのつど読む（こちらで覚え込むと、人が動かした値を踏み潰す）。これは
機械の反射で、主LLM は通らない（`ユースケース④` [D-会話減音]）。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ..core.music_rules import RESTORE_AFTER_SEC, duck, expired

logger = logging.getLogger(__name__)


async def check_music_expired(
    *, io: Any, bus: Any, state: Any, now: float, say: Callable[[str], Any]
) -> bool:
    """30 分を過ぎていたら止めて一言言う。止めたら True。"""
    if not getattr(state, "playing", False):
        return False
    started = float(getattr(state, "started_at", 0.0) or 0.0)
    if not started or not expired(started, now=now):
        return False
    await io.stop(bus)
    state.playing = False
    logger.info("音楽：30 分たったので止めた")
    say("30 分たったから、音楽を止めるね")
    return True


async def duck_while_speaking(
    *,
    io: Any,
    bus: Any,
    state: Any,
    speak: Callable[[], Awaitable[Any]],
    sleep: "Callable[[float], Awaitable[None]] | None" = None,
) -> Any:
    """声を出すあいだだけ音楽を絞る。鳴っていなければ何もしない。

    絞る前に**そのときの音量を読む**（人が変えていれば、その値へ戻す）。声が終わってから
    `RESTORE_AFTER_SEC` 秒おいて戻す——すぐ戻すと、続けて話すたびに上下してうるさい。

    **戻しは裏で予約し、待たずに返す**（出-bd ①）。以前はこの 10 秒をここで待ったので、音楽が鳴っていると
    声を 1 つ出すたびに次へ進めなかった（実機 10/10 11:05:57：17 字の聞き返しで `DIF 声 12.46 秒`）。予約中に
    次の声が来たら予約を取り消し、絞ったまま最初に読んだ基準を引き継ぐ（読み直すと絞った値を基準にしてしまう）。
    """
    base: "float | None" = None
    # **絞れなくても声は出す。** ここで例外を外へ出すと、`DIF.speak` の例外抑止に飲まれて
    # 声そのものが消える（テストが 4 件捕まえた・2026-09-21）。音楽は添え物で、声が本体である。
    with contextlib.suppress(Exception):
        pending = getattr(state, "restore_task", None)
        if isinstance(pending, asyncio.Task) and not pending.done():
            pending.cancel()
            base = state.ducked_base
        elif getattr(state, "playing", False):
            s = await io.status(bus)
            if s and s.get("playing"):
                base = float(s.get("volume", 0.5))
                await io.set_volume(bus, duck(base))
                state.ducked_base = base
                # 効いたかを後から確かめられるように残す（知-ak-ろ・2026-10-07）
                logger.info("音楽：話すあいだ音量を下げた（%.2f → %.2f）", base, duck(base))
    try:
        return await speak()
    finally:
        if base is not None:
            with contextlib.suppress(Exception):
                state.restore_task = asyncio.ensure_future(
                    _restore_later(io=io, bus=bus, state=state, base=base, sleep=sleep)
                )


async def _restore_later(
    *,
    io: Any,
    bus: Any,
    state: Any,
    base: float,
    sleep: "Callable[[float], Awaitable[None]] | None",
) -> None:
    """`RESTORE_AFTER_SEC` 秒おいて基準へ戻す（裏の仕事）。取り消されたら戻さない（次の声が絞ったまま使う）。"""
    await (sleep or asyncio.sleep)(RESTORE_AFTER_SEC)
    try:
        await io.set_volume(bus, base)
        logger.info("音楽：音量を戻した（%.2f）", base)
    except Exception:  # noqa: BLE001
        logger.warning("音楽の音量を戻せなかった", exc_info=True)
    finally:
        state.ducked_base = None


async def observe(*, io: Any, bus: Any, state: Any, record: Callable[[str], Any]) -> dict:
    """鳴っているあいだの様子を読み、曲送りと鳴り終わりを記録する（知-aa 段 2）。いまの様子を返す。

    MPRIS を読むのはここだけで、T の見張り（30 秒ごと）と反復が呼ぶ。黙って聴いているあいだの曲も
    残す（本人の決定イ）。曲は最後に書いた曲名と違うときだけ書く。止まっていたら印を下ろし（声の入口の
    門が開く）、声には出さずに記録する。**読めないときは何もしない**——機器は落ちる前提で、読めない
    ことを「止まった」とは見ない。鳴らしていなければ読まない。
    """
    if not getattr(state, "playing", False):
        return {}
    try:
        s = await io.status(bus)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        logger.warning("音楽：いまの様子を読めなかった", exc_info=True)
        return {}
    if not s or not s.get("playing"):
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
