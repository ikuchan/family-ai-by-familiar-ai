"""環-aw 第 3 段 項目 1：耳の待っている答え（`設計方針_パジュの状態` v0.15 §2.1、§3.1、決定ノ、グ、ゲ、ズ、ゼ）。

待っている答えは、答えなし、確認、申し出、聞き返しの 4 つ。持ち主は状態図で、問いの中身は状態図のデータ。新しい問いは
前の問いを上書きし、答えを使うとなしへ戻る。確認は 300 秒、聞き返しは 30 秒で切れる。申し出は窓の番号が同じあいだ
使い、窓が閉じたとき頭が何もしていないか、求めを閉じたとき窓が閉じていれば、なしへ戻る。

置く口は、データを待つ前に置き、移りはそのあと順番を待って送る。なしへ戻す移りは、自分の問いのデータのときだけ
消す。移りは打ち切りから守る（決定ズ）。
"""

from __future__ import annotations

import asyncio
import time

from familiar_agent.core.confirm_state import PendingConfirm
from familiar_agent.loop.paju_state import PajuState

TTL = 0.05


def _pc(what: str = "3 分のタイマー") -> PendingConfirm:
    return PendingConfirm(
        action="set_timer",
        tool_input={"after_minutes": 3},
        asked_at=time.time(),
        text=f"{what}でいい？",
        what=what,
    )


def _answer(sm: PajuState) -> str:
    ids = {s.id for s in sm.configuration}
    return next(
        (n for n in ("no_answer", "confirm_answer", "offer_answer", "heard_answer") if n in ids),
        "",
    )


async def _chart(**kw) -> PajuState:
    sm = PajuState(**kw)
    await sm.start()
    return sm


def test_it_starts_with_no_answer():
    async def go():
        sm = await _chart()
        assert _answer(sm) == "no_answer"
        assert sm.confirm() is None

    asyncio.run(go())


def test_asking_to_confirm_awaits_the_confirm_answer():
    async def go():
        sm = await _chart()
        pc = _pc()
        await sm.ask_confirm(pc)
        assert _answer(sm) == "confirm_answer"
        assert sm.confirm() is pc

    asyncio.run(go())


def test_a_new_question_overwrites_the_last():
    """新しい問いは前の問いを上書きする（決定ノ）。"""

    async def go():
        sm = await _chart()
        await sm.ask_confirm(_pc())
        await sm.make_offer("調べてみる？", session=1)
        assert _answer(sm) == "offer_answer"
        assert sm.confirm() is None
        second = _pc("5 分のタイマー")
        await sm.ask_confirm(second)
        await sm.ask_confirm(third := _pc("7 分のタイマー"))
        assert _answer(sm) == "confirm_answer"
        assert sm.confirm() is third

    asyncio.run(go())


def test_taking_the_confirm_returns_it_once():
    async def go():
        sm = await _chart()
        pc = _pc()
        await sm.ask_confirm(pc)
        assert await sm.take_confirm() is pc
        assert await sm.take_confirm() is None
        assert _answer(sm) == "no_answer"

    asyncio.run(go())


def test_the_confirm_lapses_after_its_time():
    """確認は 300 秒（`confirm_ttl_sec`）で切れ、なしへ戻る（決定ゲ）。切れたあとは読み手も無いと見る（決定ゼ）。"""

    async def go():
        sm = await _chart(confirm_ttl_sec=TTL)
        await sm.ask_confirm(_pc())
        await asyncio.sleep(TTL * 4)
        assert _answer(sm) == "no_answer"
        assert sm.confirm() is None
        assert await sm.take_confirm() is None

    asyncio.run(go())


def test_the_heard_check_lapses_after_its_time():
    """聞き返しは 30 秒で切れる（決定ラ、ゲ）。"""

    async def go():
        sm = await _chart(heard_ttl_sec=TTL)
        await sm.ask_heard({"action": "set_timer", "question": "タイマー？"})
        assert _answer(sm) == "heard_answer"
        await asyncio.sleep(TTL * 4)
        assert _answer(sm) == "no_answer"
        assert await sm.take_heard() is None

    asyncio.run(go())


def test_taking_the_heard_check_returns_it_once():
    async def go():
        sm = await _chart()
        check = {"action": "set_timer", "question": "タイマー？"}
        await sm.ask_heard(check)
        assert await sm.take_heard() == check
        assert await sm.take_heard() is None
        assert _answer(sm) == "no_answer"

    asyncio.run(go())


def test_the_offer_is_used_only_in_the_same_window():
    """申し出は、預けたときの窓の番号と答えた求めの窓の番号が同じときだけ使う（決定ゲ）。使っても使わなくても消す。"""

    async def go():
        sm = await _chart()
        await sm.make_offer("調べてみる？", session=3, deep={"query": "q"})
        assert await sm.take_offer(session=3) == ("調べてみる？", {"query": "q"})
        assert await sm.take_offer(session=3) is None
        assert _answer(sm) == "no_answer"
        await sm.make_offer("調べてみる？", session=3)
        assert await sm.take_offer(session=4) is None
        assert _answer(sm) == "no_answer"

    asyncio.run(go())


def test_the_offer_ends_when_the_window_lapses_while_idle():
    """窓が閉じたとき頭が何もしていなければ、申し出はなしへ戻る。"""

    async def go():
        sm = await _chart()
        await sm.fire("window_opened")
        await sm.make_offer("調べてみる？", session=1)
        await sm.fire("window_lapsed")
        assert _answer(sm) == "no_answer"
        assert await sm.take_offer(session=1) is None

    asyncio.run(go())


def test_a_reply_in_the_window_still_finds_the_offer_after_the_window_lapses():
    """窓の中で受けた返事の判断より前に窓が閉じても、求めの途中なら申し出は残る（9 秒目の返事・2026-11-17 本人の決定）。
    求めを閉じたとき窓が閉じていれば、なしへ戻る。"""

    async def go():
        sm = await _chart()
        await sm.fire("window_opened")
        await sm.make_offer("調べてみる？", session=1)
        await sm.fire("request_opened")  # 窓の中の「うん」で求めが開いた
        await sm.fire("window_lapsed")  # 判断の前に窓が閉じた
        assert _answer(sm) == "offer_answer"
        await sm.fire("request_closed")
        assert _answer(sm) == "no_answer"

    asyncio.run(go())


def test_the_offer_stays_when_the_request_closes_in_an_open_window():
    """申し出で終わった返事を言い終えると窓が開く。求めを閉じても窓が開いていれば、申し出は答えを待つ。"""

    async def go():
        sm = await _chart()
        await sm.fire("request_opened")
        await sm.make_offer("調べてみる？", session=1)
        await sm.fire("window_opened")
        await sm.fire("request_closed")
        assert _answer(sm) == "offer_answer"

    asyncio.run(go())


def test_the_data_is_placed_before_waiting_for_the_turn():
    """**決定ズの 1**：データは待つ前に置く。順番を待つあいだに走った 2 つめの `set_timer` も、確認を見て落ちる。"""

    async def go():
        sm = await _chart()
        pc = _pc()
        async with sm._turn():
            task = asyncio.create_task(sm.ask_confirm(pc))
            await asyncio.sleep(0)
            assert sm.confirm() is pc  # 状態はまだ答えなしでも、読み手はデータで確認を見る
            assert _answer(sm) == "no_answer"
        await task
        assert _answer(sm) == "confirm_answer"

    asyncio.run(go())


def test_a_lapse_does_not_clear_a_newer_question():
    """**決定ズの 2**：先に並んだ聞き返しの切れは、あとから置いた確認のデータを消さない。"""

    async def go():
        sm = await _chart(heard_ttl_sec=TTL)
        await sm.ask_heard({"action": "set_timer", "question": "タイマー？"})
        async with sm._turn():
            await asyncio.sleep(TTL * 4)  # 切れる仕事は順番を待っている
            task = asyncio.create_task(sm.ask_confirm(pc := _pc()))
            await asyncio.sleep(0)
        await task
        await asyncio.sleep(TTL)
        assert sm.confirm() is pc
        assert _answer(sm) == "confirm_answer"

    asyncio.run(go())


def test_a_cancelled_ask_still_moves_the_state():
    """**決定ズの 3**：順番を待つあいだに道具の仕事が止められても、移りは最後まで送る（データと状態をずらさない）。"""

    async def go():
        sm = await _chart()
        async with sm._turn():
            task = asyncio.create_task(sm.ask_confirm(pc := _pc()))
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
        await asyncio.sleep(0.01)
        assert task.cancelled()
        assert sm.confirm() is pc
        assert _answer(sm) == "confirm_answer"

    asyncio.run(go())


def test_the_confirm_prompt_time_lives_with_the_confirm():
    """決まった促しを最後に言った時刻は、確認の中身（決定グ）。新しい確認を置くと消える。"""

    async def go():
        sm = await _chart()
        await sm.ask_confirm(_pc())
        assert sm.confirm_prompted_at() is None
        sm.note_confirm_prompt(123.0)
        assert sm.confirm_prompted_at() == 123.0
        await sm.ask_confirm(_pc("5 分のタイマー"))
        assert sm.confirm_prompted_at() is None

    asyncio.run(go())


def test_the_log_line_names_the_awaited_answer():
    async def go():
        sm = await _chart()
        await sm.ask_confirm(_pc())
        assert "答え=確認" in sm.describe()

    asyncio.run(go())


def test_dropping_the_offer_leaves_other_questions_alone():
    """申し出でない返事を話したら、前の申し出だけを消す（出-dj）。確認や聞き返しには触らない。"""

    async def go():
        sm = await _chart()
        await sm.make_offer("調べてみる？", session=1)
        assert sm.offer_waiting()
        await sm.drop_offer()
        assert not sm.offer_waiting() and _answer(sm) == "no_answer"
        await sm.ask_confirm(pc := _pc())
        await sm.drop_offer()  # **反証の側**：確認は消さない
        assert sm.confirm() is pc and _answer(sm) == "confirm_answer"
        assert not sm.offer_waiting()

    asyncio.run(go())


def test_the_offer_waiting_reads_the_data_before_the_turn():
    """決定ズの 1：申し出を待っているかも、状態でなくデータを読む。"""

    async def go():
        sm = await _chart()
        async with sm._turn():
            task = asyncio.create_task(sm.make_offer("調べてみる？", session=1))
            await asyncio.sleep(0)
            assert sm.offer_waiting() and _answer(sm) == "no_answer"
        await task

    asyncio.run(go())


def test_a_chart_that_cannot_start_still_keeps_the_data():
    """**反証の側**：状態図が初めの状態に入れなくても、置く口は例外を出さず、データは置く（読み手はデータを読む・決定ズの 1）。"""
    from unittest.mock import AsyncMock, patch

    async def go():
        sm = PajuState()
        with patch.object(sm, "start", AsyncMock(side_effect=RuntimeError("start"))):
            await sm.make_offer("調べようか？", session=1)
        assert sm.offer_waiting()

    asyncio.run(go())
