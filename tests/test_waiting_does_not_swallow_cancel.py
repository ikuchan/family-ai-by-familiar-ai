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


# ── 常駐のループ（環-aa・2026-10-05）────────────────────────────────────────────
#
# 駆動体の外でも、止めるときにキャンセルされ続ける常駐のループが `wait_for` で待っていた。キャンセルが消えると、
# アプリの終了がそこで待ち続ける。在席センサは知-ai で 3 秒ごとになり、待ちが終わる回数が 10 倍になった。
# ファイルには終了時の片付け（1 回きりの待ち）の `wait_for` も残るので、見張るのはループの関数の本文だけ。


def _function_source(path: str, name: str) -> str:
    import ast

    text = (SRC / path).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(text, node) or ""
    raise AssertionError(f"{path} に {name} が無い")


@pytest.mark.parametrize(
    "path,name",
    [
        ("occupancy_sensor.py", "_run"),  # 在席センサの見回り
        ("gui.py", "_process_queue"),  # 画面の入力の待ち
        ("main.py", "_next_input"),  # CUI の入力の待ち
    ],
)
def test_the_resident_loops_do_not_use_wait_for(path, name):
    body = _function_source(path, name)
    assert "asyncio.wait_for(" not in body
    assert "wait_within(" in body
