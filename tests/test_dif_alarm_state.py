"""環-aw 第 6 段 項目 1：知らせの音を状態図の口の領域にする（`設計方針_パジュの状態` §3.5・決定ニ）。

知らせの音は、タイマーとアラームが鳴らす音の領域で、状態は止まっている（初め）と鳴っている。持ち主は dif の鳴らしている
仕事で、状態図はその写し（入口と同じ形）。写すのは自律機構の輪（0.5 秒ごと・`copy_ring`）だけ。いま鳴っているか
（`DIF.ringing`）は、いままでどおり鳴らしている仕事を見る（止めたらすぐ偽、音は状態図を待たない）。
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

from familiar_agent.io import dif as dif_mod
from familiar_agent.loop.paju_state import PajuState
from familiar_agent.loop.tonic import Tonic


def _fake_wav(monkeypatch, gate: "asyncio.Event | None" = None) -> list[float]:
    played: list[float] = []

    async def _play(path, gain):  # noqa: ANN001
        played.append(gain)
        if gate is not None:
            await gate.wait()
        await asyncio.sleep(0)
        return True

    monkeypatch.setattr(dif_mod, "_play_wav", _play)
    return played


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


def test_the_chart_has_an_alarm_region_that_starts_quiet():
    async def go():
        sm = PajuState()
        await sm.start()
        assert sm.is_ringing() is False
        await sm.fire("ring_began")
        assert sm.is_ringing() is True
        assert "知らせの音=鳴っている" in sm.describe()
        await sm.fire("ring_ended")
        assert sm.is_ringing() is False
        assert "知らせの音=止まっている" in sm.describe()

    asyncio.run(go())


def test_copy_ring_follows_the_owner_and_logs_only_changes(caplog):
    async def go():
        sm = PajuState()
        owner = {"on": True}
        with caplog.at_level(logging.INFO, logger="familiar_agent.loop.paju_state"):
            await sm.copy_ring(lambda: owner["on"])
            await sm.copy_ring(lambda: owner["on"])  # 同じなら何もしない
            assert sm.is_ringing()
            owner["on"] = False
            await sm.copy_ring(lambda: owner["on"])
            assert not sm.is_ringing()
        lines = [r.getMessage() for r in caplog.records if "知らせの音を写した" in r.getMessage()]
        assert lines == [
            "状態図 知らせの音を写した：鳴っている",
            "状態図 知らせの音を写した：止まっている",
        ]

    asyncio.run(go())


def test_a_broken_owner_inside_the_turn_is_warned_and_does_not_raise(caplog):
    calls = {"n": 0}

    def flaky() -> bool:
        calls["n"] += 1
        if calls["n"] > 1:  # 順番の外の 1 回目は読めて、順番の中の 2 回目で壊れる
            raise RuntimeError("読めない")
        return True

    async def go():
        sm = PajuState()
        await sm.start()
        with caplog.at_level(logging.WARNING):
            await sm.copy_ring(flaky)
        assert any("知らせの音を写せなかった" in r.getMessage() for r in caplog.records)
        assert not sm.is_ringing()

    asyncio.run(go())


def test_the_tonic_copies_the_dif_ring_each_turn(monkeypatch):
    """自律機構の輪が、dif の鳴らしている仕事を写す。鳴らし直し・止めたあとも、次の輪で合う。音は状態図を待たない。"""
    gate = asyncio.Event()
    played = _fake_wav(monkeypatch, gate)

    async def go():
        sm = PajuState()
        await sm.start()
        d = dif_mod.DIF()
        agent = SimpleNamespace(paju_state=sm, timer_ringing=lambda: d.ringing)
        t = Tonic(SimpleNamespace(), agent=agent)
        async with sm._turn():  # 状態図の順番がふさがっていても、音は鳴り始める
            d.ring(seconds=30)
            await _settle()
            assert played and d.ringing
        await t._copy_ring()
        assert sm.is_ringing()
        d.ring(seconds=30)  # 鳴っているあいだに次の音（アラームの最中にタイマー）
        await _settle()
        await t._copy_ring()
        assert sm.is_ringing()
        d.stop_ring()
        assert not d.ringing  # 持ち主は止めたらすぐ偽
        await _settle()
        await t._copy_ring()
        assert not sm.is_ringing()

    asyncio.run(go())
