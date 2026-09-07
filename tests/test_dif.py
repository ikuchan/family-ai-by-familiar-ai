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


def _dif(*, tts=None, search=None, fetch=None) -> DIF:
    """**口は agent を知らない。** 要るのは声と2つの道具だけである。"""
    return DIF(tts=tts, search=search, fetch=fetch)


# ── 声 ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_speaking_goes_through_to_the_speaker():
    tts = MagicMock()
    tts.call = AsyncMock(return_value=("Said: はい", None))
    await _dif(tts=tts).speak("はい")
    tts.call.assert_awaited_once_with("say", {"text": "はい"})


@pytest.mark.asyncio
async def test_a_missing_speaker_is_not_an_error():
    """機器は落ちる前提のもの。声が出せなくてもターンごと壊さない。"""
    await _dif(tts=None).speak("はい")


@pytest.mark.asyncio
async def test_a_speaker_that_throws_is_swallowed():
    tts = MagicMock()
    tts.call = AsyncMock(side_effect=OSError("鳴らない"))
    await _dif(tts=tts).speak("はい")


def test_it_knows_whether_the_synthesiser_understands_tags():
    """担い手が何を解するかを知っているのは DIF である（`根拠台帳` §9）。"""
    tts = MagicMock()
    tts.understands_tags = True
    assert _dif(tts=tts).understands_tags is True
    assert _dif(tts=None).understands_tags is False


# ── 調べもの ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_search_is_dispatched_and_reports_whether_it_flew():
    """2つ目の返り値は**背景タスクを作ったか**。作らずに帰る経路が3つある。"""
    search = MagicMock()
    search.dispatch = AsyncMock(return_value=("投げた", True))
    got = await _dif(search=search).lookup("search_deferred", {"query": "明日の天気"})
    assert got == ("投げた", True)
    search.dispatch.assert_awaited_once_with({"query": "明日の天気"})


@pytest.mark.asyncio
async def test_a_fetch_goes_to_the_other_tool():
    fetch = MagicMock()
    fetch.dispatch = AsyncMock(return_value=("読んだ", True))
    search = MagicMock()
    search.dispatch = AsyncMock()
    await _dif(search=search, fetch=fetch).lookup("fetch_deferred", {"url": "x"})
    fetch.dispatch.assert_awaited_once()
    search.dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failing_lookup_raises_so_the_caller_can_close_the_intent():
    """**ここでは畳まない。** 開いた意図を閉じるのは呼び手の仕事で、口が飲むと
    飛行中の数が合わなくなる。"""
    search = MagicMock()
    search.dispatch = AsyncMock(side_effect=OSError("繋がらない"))
    with pytest.raises(OSError):
        await _dif(search=search).lookup("search_deferred", {})


@pytest.mark.asyncio
async def test_cancellation_is_not_swallowed():
    search = MagicMock()
    search.dispatch = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        await _dif(search=search).lookup("search_deferred", {})


# ── ループが口を通ること ────────────────────────────────────────────────────


def test_the_port_does_not_hold_the_agent():
    """口が神オブジェクトを丸ごと持たないこと。要るのは声と2つの道具だけである。

    `agent` を持てば、口は 89 個の属性すべてに手が届く。届く必要のないものへ届く形は、
    口を1枚挟んだ意味を消す。
    """
    import inspect

    params = set(inspect.signature(DIF.__init__).parameters)
    assert "agent" not in params
    assert {"tts", "search", "fetch"} <= params


def test_the_loop_reaches_for_a_device_only_when_it_wires_the_port():
    """移し終わったことを、旧い掴み方の数で見る。

    口へ渡すために名前を1度書くのは配線である。**2度目からが掴み直し**で、それが
    増えると口を1枚挟んだ意味が消える。呼び出しの場所で機器を掴んでいないことを、
    「1つにつき1回だけ」で見る。
    """
    from pathlib import Path

    src = Path(__file__).parent.parent / "src/familiar_agent/loop/event_loop.py"
    text = src.read_text(encoding="utf-8")
    for name in ("agent._tts", "agent._deferred_search", "agent._deferred_fetch"):
        assert text.count(name) <= 1, f"{name} を呼び出しの場所で掴み直している"
