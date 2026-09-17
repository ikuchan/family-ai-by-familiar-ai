"""掛ける前に確かめる（`TIMER_CONFIRM`・段 2）と、同時に 1 本（フラグに関係ない規則・段 3）。

知-o・2026-09-18・`設計方針_タイマー` v0.3 §10・§11。
"""

from __future__ import annotations

import asyncio

import pytest

from tests.test_timer_tool import _tool


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    monkeypatch.setenv("TIMER_SILENCE", "true")
    monkeypatch.setenv("TIMER_MIC_CLOSE", "true")
    monkeypatch.setenv("TIMER_CONFIRM", "true")


# ── 段 2：確かめる ──────────────────────────────────────────────────────────


def test_a_timer_asks_first_when_confirm_is_on():
    t, store, _ = _tool()
    text, ok = asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}))
    assert ok and store.active() == []  # まだ掛けていない
    assert "3 分" in text and "黙って" in text and "聞かない" in text and "confirmed=true" in text


def test_the_question_matches_the_flags(monkeypatch):
    monkeypatch.setenv("TIMER_SILENCE", "false")
    monkeypatch.setenv("TIMER_MIC_CLOSE", "false")
    t, _, _ = _tool()
    text, _ = asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}))
    assert "黙" not in text and "聞かない" not in text and "3 分" in text


def test_confirmed_sets_the_timer():
    t, store, _ = _tool()
    text, ok = asyncio.run(
        t.call("set_timer", {"after_minutes": 3, "label": "パスタ", "confirmed": True})
    )
    assert ok and len(store.active()) == 1 and "id=1" in text


def test_confirm_off_sets_at_once(monkeypatch):
    monkeypatch.setenv("TIMER_CONFIRM", "false")
    t, store, _ = _tool()
    asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}))
    assert len(store.active()) == 1


# ── 段 3：同時に 1 本 ───────────────────────────────────────────────────────


def test_only_one_timer_at_a_time(monkeypatch):
    monkeypatch.setenv("TIMER_CONFIRM", "false")
    t, store, _ = _tool()
    asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}))
    text, ok = asyncio.run(t.call("set_timer", {"after_minutes": 5, "label": "お茶"}))
    assert not ok and "パスタ" in text and "止める" in text and len(store.active()) == 1


def test_the_running_timer_is_reported_before_any_confirmation():
    """確認だけして掛からない、を避ける：動いていればまず「動いている」を返す。"""
    t, store, _ = _tool()
    asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ", "confirmed": True}))
    text, ok = asyncio.run(t.call("set_timer", {"after_minutes": 5, "label": "お茶"}))
    assert not ok and "パスタ" in text and "confirmed" not in text


def test_a_stopwatch_can_run_beside_a_timer_but_only_one_stopwatch(monkeypatch):
    monkeypatch.setenv("TIMER_CONFIRM", "false")
    t, store, _ = _tool()
    asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}))
    _, ok1 = asyncio.run(t.call("start_stopwatch", {"label": "ランニング"}))
    text, ok2 = asyncio.run(t.call("start_stopwatch", {"label": "散歩"}))
    assert ok1 and not ok2 and "ランニング" in text and len(store.active()) == 2
