"""返事とつなぎは窓の中でだけ出し、出したら窓を延ばす（出-as 段 4・2026-09-26・`設計方針_話していいかの決まり` §2.3）。

- 会話の求めの返事を出す時点で**窓が切れていたら、話さずに独り言にする**（本人の決定）。受けた時点では
  会話だったが、1 分を過ぎてから答えても相手はもう聞いていない。
- 返事を出したら、そこから 1 分へ延ばす（「パジュ、明日の天気は？」→ 返事 →「じゃあ傘いる？」が続く）。
- つなぎも同じ：窓が切れていれば言わない。言ったら延ばす。
- 情動・機器の求めは窓で決めない（別の決まり）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def _ip(kind="発話", *, window_open=True, now=1000.0, monkeypatch=None):
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip._req.trigger_kind = kind
    ip._req.fired_axis = "bond"  # 情動なら話しかける軸（出-as 段 8：話すのは bond・esteem だけ）
    ip._req.request_text = "おはよう"
    ip._delivery_block_reason = lambda: ""
    ip._dif = MagicMock()
    ip._dif.speak = AsyncMock()
    ip._speak_filler_in_background = MagicMock()
    ip._oif_write_filler = AsyncMock()
    if window_open:
        ip._wake.open(now - 15.0)  # 15 秒前に開いた（残り 15 秒）
    else:
        ip._wake.open(now - 45.0)  # 45 秒前に開いて切れた
    monkeypatch.setattr("familiar_agent.loop.event_loop.time.monotonic", lambda: now)
    return a, ip


def test_a_reply_inside_the_window_is_spoken_and_extends_it(monkeypatch):
    a, ip = _ip(monkeypatch=monkeypatch)
    spoken, outcome = asyncio.run(ip._speak("晴れだよ"))
    assert (spoken, outcome) == ("晴れだよ", "発話")
    ip._dif.speak.assert_awaited_once()
    assert ip._wake.is_open(1029.0)  # 返事から 30 秒


def test_a_reply_after_the_window_closed_is_kept_as_a_monologue(monkeypatch):
    a, ip = _ip(window_open=False, monkeypatch=monkeypatch)
    spoken, outcome = asyncio.run(ip._speak("晴れだよ"))
    assert (spoken, outcome) == ("晴れだよ", "独白")
    ip._dif.speak.assert_not_awaited()
    assert not ip._wake.is_open(1000.0)  # 延ばさない


def test_a_filler_inside_the_window_is_said_and_extends_it(monkeypatch):
    a, ip = _ip(monkeypatch=monkeypatch)
    a._oif.write = AsyncMock(return_value="obs-f")
    asyncio.run(ip._say_filler("ちょっと待ってね"))
    ip._speak_filler_in_background.assert_called_once()
    assert ip._wake.is_open(1029.0)


def test_a_filler_after_the_window_closed_is_not_said(monkeypatch):
    a, ip = _ip(window_open=False, monkeypatch=monkeypatch)
    asyncio.run(ip._say_filler("ちょっと待ってね"))
    ip._speak_filler_in_background.assert_not_called()


def test_affect_is_not_governed_by_the_window(monkeypatch):
    """情動の発話は窓で決めない（§2.1）。窓が切れていても出口の門のとおりに出る。"""
    a, ip = _ip("情動", window_open=False, monkeypatch=monkeypatch)
    spoken, outcome = asyncio.run(ip._speak("ねえねえ"))
    assert outcome == "発話"
