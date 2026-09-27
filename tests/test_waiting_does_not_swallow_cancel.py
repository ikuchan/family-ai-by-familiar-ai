"""時間切れ付きで待つ口は、キャンセルを握りつぶさない（環-aa・2026-09-27）。

Python 3.11 の `asyncio.wait_for` は、中の処理が終わるのとキャンセルが同時に来ると、キャンセルを消して結果を返す
（3.12 で作り直されて直った）。駆動体は反復のたびに調停の返りを `wait_for` で待っていたので、`close()`（試験の終わり・
停止・終了）がキャンセルした瞬間に返りが届くと、駆動体は「キャンセル中」のまま次の待ち（`_take_trigger`）に入って
止まった（全体テストで `test_event_loop.py` がまれに 90 秒で止まった原因）。`asyncio.timeout()` なら消えない。
"""

from __future__ import annotations

import asyncio
import pathlib

import pytest

from familiar_agent.core.aio import wait_within

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "familiar_agent"


def _race():
    """中の処理が終わるのと同じ瞬間にキャンセルする。キャンセルが消えれば、次の待ちで止まる。"""

    async def go():
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        reached: list = []

        async def driver():
            await wait_within(asyncio.shield(fut), 5.0)
            reached.append("次の待ち")
            await asyncio.sleep(3600)

        t = loop.create_task(driver())
        await asyncio.sleep(0)
        fut.set_result("返り")
        t.cancel()
        await asyncio.sleep(0.05)
        stuck = not t.done()
        t.cancel()
        try:
            await t
        except BaseException:  # noqa: BLE001
            pass
        return stuck, reached

    return asyncio.run(go())


def test_cancel_is_not_swallowed_when_the_result_arrives_at_the_same_moment():
    stuck, reached = _race()
    assert not stuck and reached == []


def test_it_still_times_out():
    async def go():
        await wait_within(asyncio.sleep(1.0), 0.02)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(go())


def test_it_returns_the_result():
    async def go():
        async def ok():
            return 7

        return await wait_within(ok(), 1.0)

    assert asyncio.run(go()) == 7


@pytest.mark.parametrize(
    "path", ["loop/arbiter.py", "loop/event_loop.py", "backends/jev.py", "core/jev_judges.py"]
)
def test_the_driver_paths_do_not_use_wait_for(path):
    assert "asyncio.wait_for(" not in (SRC / path).read_text(encoding="utf-8")
