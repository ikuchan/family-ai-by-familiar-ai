"""外部機器接続 DIF：ループと外の機械のあいだの唯一の口（環-e-は・段1）。

設計（`設計図` ③-2）は出入り口を4つ（IIF・DIF・AIF・OIF）と定め、この4つ以外に
コンポーネントどうしが直接つながる線を置かない。ところが**外部の機械へは口が無く**、
`loop/event_loop.py` がスピーカー・カメラ・調べものの道具を直接掴んでいた。

**挙動は変えない。** 返り値の形も、例外の畳み方も、順序もそのままで、口は転送し、
通ったものを debug に残すだけである。

段1は**声と調べもの**だけを通す。知覚（`see`・`look`）は 知-c が実機で挙動を追っている
最中なので、そこが済むまで触らない（同じ箇所を切り出しと機能変更の両方で触ると、
どちらで挙動が変わったのか切り分けられなくなる）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.io.dif import DIF


def _agent(*, tts=None, search=None, fetch=None):
    a = MagicMock()
    a._tts = tts
    a._deferred_search = search
    a._deferred_fetch = fetch
    return a


# ── 声 ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_speaking_goes_through_to_the_speaker():
    tts = MagicMock()
    tts.call = AsyncMock(return_value=("Said: はい", None))
    await DIF(_agent(tts=tts)).speak("はい")
    tts.call.assert_awaited_once_with("say", {"text": "はい"})


@pytest.mark.asyncio
async def test_a_missing_speaker_is_not_an_error():
    """機器は落ちる前提のもの。声が出せなくてもターンごと壊さない。"""
    await DIF(_agent(tts=None)).speak("はい")


@pytest.mark.asyncio
async def test_a_speaker_that_throws_is_swallowed():
    tts = MagicMock()
    tts.call = AsyncMock(side_effect=OSError("鳴らない"))
    await DIF(_agent(tts=tts)).speak("はい")


def test_it_knows_whether_the_synthesiser_understands_tags():
    """担い手が何を解するかを知っているのは DIF である（`根拠台帳` §9）。"""
    tts = MagicMock()
    tts.understands_tags = True
    assert DIF(_agent(tts=tts)).understands_tags is True
    assert DIF(_agent(tts=None)).understands_tags is False


# ── 調べもの ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_search_is_dispatched_and_reports_whether_it_flew():
    """2つ目の返り値は**背景タスクを作ったか**。作らずに帰る経路が3つある。"""
    search = MagicMock()
    search.dispatch = AsyncMock(return_value=("投げた", True))
    got = await DIF(_agent(search=search)).lookup("search_deferred", {"query": "明日の天気"})
    assert got == ("投げた", True)
    search.dispatch.assert_awaited_once_with({"query": "明日の天気"})


@pytest.mark.asyncio
async def test_a_fetch_goes_to_the_other_tool():
    fetch = MagicMock()
    fetch.dispatch = AsyncMock(return_value=("読んだ", True))
    search = MagicMock()
    search.dispatch = AsyncMock()
    await DIF(_agent(search=search, fetch=fetch)).lookup("fetch_deferred", {"url": "x"})
    fetch.dispatch.assert_awaited_once()
    search.dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failing_lookup_raises_so_the_caller_can_close_the_intent():
    """**ここでは畳まない。** 開いた意図を閉じるのは呼び手の仕事で、口が飲むと
    飛行中の数が合わなくなる。"""
    search = MagicMock()
    search.dispatch = AsyncMock(side_effect=OSError("繋がらない"))
    with pytest.raises(OSError):
        await DIF(_agent(search=search)).lookup("search_deferred", {})


@pytest.mark.asyncio
async def test_cancellation_is_not_swallowed():
    search = MagicMock()
    search.dispatch = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        await DIF(_agent(search=search)).lookup("search_deferred", {})


# ── ループが口を通ること ────────────────────────────────────────────────────


def test_the_loop_no_longer_holds_the_devices_itself():
    """移し終わったことを、旧い掴み方が0件であることで見る。"""
    from pathlib import Path

    src = Path(__file__).parent.parent / "src/familiar_agent/loop/event_loop.py"
    text = src.read_text(encoding="utf-8")
    for old in ("agent._tts", "agent._deferred_search", "agent._deferred_fetch"):
        assert old not in text, f"{old} がループに残っている"
