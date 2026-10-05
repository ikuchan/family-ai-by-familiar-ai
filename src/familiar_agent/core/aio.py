"""時間切れ付きで待つ口（環-aa・2026-09-27）。

Python 3.11 の `asyncio.wait_for` は、中の処理が終わるのとキャンセルが同時に来ると、キャンセルを消して結果を返す
（3.12 で作り直されて直った）。駆動体がこれで待っていると、`close()` や停止のキャンセルが消え、駆動体は
「キャンセル中」のまま次の待ちで止まる。3.11 以上は `asyncio.timeout()` で待つ（キャンセルを消さない）。
3.10 には `asyncio.timeout` が無いので `wait_for` のまま（この握りつぶしは残る）。

駆動体のほか、止めるときにキャンセルされ続ける常駐のループ（在席センサの見回り・画面の入力・CUI の入力）もこれで
待つ（2026-10-05）。終了時の片付けの 1 回きりの待ちは `wait_for` のまま。
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, TypeVar

T = TypeVar("T")


async def wait_within(aw: "Awaitable[T]", timeout: float) -> T:
    """`aw` を最大 `timeout` 秒待つ。過ぎれば `asyncio.TimeoutError`。キャンセルは握りつぶさない。"""
    timeout_cm = getattr(asyncio, "timeout", None)
    if timeout_cm is None:  # pragma: no cover  Python 3.10
        return await asyncio.wait_for(aw, timeout=timeout)
    async with timeout_cm(timeout):
        return await aw
