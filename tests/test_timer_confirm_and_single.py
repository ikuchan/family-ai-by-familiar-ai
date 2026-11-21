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
    assert "3 分" in text and "黙って" in text and "聞かない" in text
    assert "confirmed" not in text  # 掛け直しは LLM の仕事ではない（出-y・機械の confirm）


def test_the_question_matches_the_flags(monkeypatch):
    monkeypatch.setenv("TIMER_SILENCE", "false")
    monkeypatch.setenv("TIMER_MIC_CLOSE", "false")
    t, _, _ = _tool()
    text, _ = asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}))
    assert "黙" not in text and "聞かない" not in text and "3 分" in text


def test_confirmed_sets_the_timer():
    """`confirmed` は機械だけが立てるキーワード引数（`agent.resolve_confirm`）。"""
    t, store, _ = _tool()
    text, ok = asyncio.run(
        t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}, confirmed=True)
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


def test_a_paused_timer_also_blocks_another_one(monkeypatch):
    """一時停止中のタイマーがあっても、次は掛けられない（2026-11-21 本人「黙るタイマーは複数かけられないようにして」）。

    一時停止しても鳴る時刻（`due`）は残るので、一時停止中も「動いている」と数える。数えないと、止めておいたタイマーを再開した
    とき 2 本が同時に動き、黙る依頼は片方のタイマーにしか結びつかない。黙るのはタイマーだけ（アラームは黙らない）。
    """
    monkeypatch.setenv("TIMER_CONFIRM", "false")
    t, store, _ = _tool()
    asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}))
    asyncio.run(t.call("pause_timer", {"id": "all"}))
    text, ok = asyncio.run(t.call("set_timer", {"after_minutes": 5, "label": "お茶"}))
    assert not ok and "パスタ" in text and "止める" in text and len(store.active()) == 1


def test_the_running_timer_is_reported_before_any_confirmation():
    """確認だけして掛からない、を避ける：動いていればまず「動いている」を返す。"""
    t, store, _ = _tool()
    asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}, confirmed=True))
    text, ok = asyncio.run(t.call("set_timer", {"after_minutes": 5, "label": "お茶"}))
    assert not ok and "パスタ" in text and "確かめ" not in text
