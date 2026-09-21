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
    """
    base: "float | None" = None
    # **絞れなくても声は出す。** ここで例外を外へ出すと、`DIF.speak` の例外抑止に飲まれて
    # 声そのものが消える（テストが 4 件捕まえた・2026-09-21）。音楽は添え物で、声が本体である。
    with contextlib.suppress(Exception):
        if getattr(state, "playing", False):
            s = await io.status(bus)
            if s and s.get("playing"):
                base = float(s.get("volume", 0.5))
                await io.set_volume(bus, duck(base))
    try:
        return await speak()
    finally:
        if base is not None:
            with contextlib.suppress(Exception):
                await (sleep or asyncio.sleep)(RESTORE_AFTER_SEC)
                await io.set_volume(bus, base)
