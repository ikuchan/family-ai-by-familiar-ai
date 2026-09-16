"""列から取り出す口は `_take_trigger` の 1 つ（2026-09-16 実機 12:40 の詰まり）。

「おやすみなさい」（会話入力・呼び手が Future で待つ）が 88 秒返らず、STT の入力が 6 件
溜まった。ログで確定した穴は 3 つ。

- **A**：`_take_trigger` が保留箱を列より先に返す。列で待つ会話入力より、保留箱の入室が先に
  求めを始めた（規則は 会話入力 ＞ 機器 ＞ 情動 なのに）。
- **B-1**：`_intake` が列を種類を見ずに空にする。3 本のキューだったころの「QC を drain」が
  `a57cf94` で名前だけ置き換わって残っていた。入室の求めの取込が会話入力を完了として取った。
- **B-2**：取込は、対応する意図の無い物を無言で捨てる。会話入力の Future は誰にも触られず、
  GUI の `await` が永久に返らなかった。

直し：会話入力は保留箱より常に先（調査中でも）。取込は列に触らず完了箱だけを見る。意図の
無い完了は warning、万一 会話入力 が来たら error にして Future へ例外を返す（何も溜めない・
起きたら必ず見える）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock


from familiar_agent.loop.event_loop import InformationProcessing, Trigger
from familiar_agent.loop.request import Lookup


def _ip():
    ip = InformationProcessing(MagicMock())
    ip._write_version = AsyncMock(return_value="v")
    return ip


def _run(coro):
    """空の列を待ち続ける（＝落ちた側）ときにテストが固まらないよう、締切を置く。"""

    async def bounded():
        return await asyncio.wait_for(coro, timeout=2.0)

    return asyncio.run(bounded())


# ── A ────────────────────────────────────────────────────────────────────────


def test_an_utterance_in_the_queue_beats_a_device_already_held():
    """打ち切りで保留箱へ回った入室より、列で待つ人の言葉が先。"""
    ip = _ip()

    async def scenario():
        ip._held.append(Trigger(kind="機器", query="入室", result="誰か が来た"))
        ip._triggers.put_nowait(Trigger(kind="会話入力", query="おやすみなさい"))
        return [(await ip._take_trigger()).kind for _ in range(2)]

    assert _run(scenario()) == ["会話入力", "機器"]


def test_a_held_utterance_is_taken_even_while_a_lookup_is_in_flight():
    """保留箱に会話入力があれば、調査中でもそれを返す（保留箱で腐らない）。"""
    ip = _ip()

    async def scenario():
        ip._req.lookups = [Lookup(index=1, action="see", query="q", generation=0)]
        ip._held.append(Trigger(kind="機器", query="入室", result="誰か が来た"))
        ip._held.append(Trigger(kind="会話入力", query="ねえ"))
        got = await ip._take_trigger()
        return got.kind, [t.kind for t in ip._held]

    assert _run(scenario()) == ("会話入力", ["機器"])


def test_without_an_utterance_the_held_device_comes_out_when_nothing_is_in_flight():
    ip = _ip()

    async def scenario():
        ip._held.append(Trigger(kind="機器", query="入室", result="誰か が来た"))
        return (await ip._take_trigger()).kind

    assert _run(scenario()) == "機器"


# ── B-1 ──────────────────────────────────────────────────────────────────────


def test_intake_leaves_the_queue_alone():
    """取込は完了箱だけを見る。列の会話入力はそのまま残る。"""
    ip = _ip()

    async def scenario():
        ip._req.lookups = [Lookup(index=1, action="recall", query="q", generation=0)]
        ip._drained_completions.append(Trigger(kind="完了", query="q", result="r"))
        ip._triggers.put_nowait(Trigger(kind="会話入力", query="おやすみなさい"))
        n, _ = await ip._intake()
        return n, ip._triggers.qsize(), ip._req.lookups[0].result

    n, left, result = _run(scenario())
    assert n == 1 and left == 1 and result == "r"


# ── B-2 ──────────────────────────────────────────────────────────────────────


def test_a_completion_without_an_open_intent_is_logged_not_silently_dropped(caplog):
    ip = _ip()

    async def scenario():
        ip._drained_completions.append(Trigger(kind="完了", query="消えた意図", result="r"))
        with caplog.at_level("WARNING", logger="familiar_agent.loop.event_loop"):
            await ip._intake()

    _run(scenario())
    assert any(
        "対応する意図が無い" in r.message and "消えた意図" in r.message for r in caplog.records
    )


def test_an_utterance_that_reaches_intake_fails_loudly_and_frees_the_caller(caplog):
    """起きないはずのこと。静かに戻さず、error に残して待ち手に例外を返す（何も溜めない）。"""
    ip = _ip()

    async def scenario():
        fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        ip._drained_completions.append(Trigger(kind="会話入力", query="ねえ", future=fut))
        with caplog.at_level("ERROR", logger="familiar_agent.loop.event_loop"):
            await ip._intake()
        return fut

    fut = _run(scenario())
    assert fut.done() and isinstance(fut.exception(), RuntimeError)
    assert any(r.levelname == "ERROR" and "会話入力" in r.message for r in caplog.records)
    assert ip._held == []
