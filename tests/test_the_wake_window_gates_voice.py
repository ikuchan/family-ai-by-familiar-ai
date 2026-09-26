"""声は、ウェイクワードの窓の中でだけ会話として受ける（出-as 段 3・2026-09-26・`設計方針_話していいかの決まり` v0.1 §2.3）。

家族どうしの話・食事のあいさつ・幻聴の定型文にまで返事をしていた（実機 2026-09-26 朝）。

- 声：名前があれば窓を開けて受ける。窓が開いていれば延ばして受ける。どちらでもなければ**捨てる**
  （記録しない・ログだけ）。**カメラに誰も映っていなくても**、窓の中なら受ける。
- キーボード：声と同じ。名前で窓を開ける（出-au 段 1-2 で改めた）。
- タイマーの操作の言葉（「止めて」など）にも名前が要る（出-au 段 1-2 で改めた）。
- 捨てた声で、しかも誰も映っていなければ、見回したくなる押し上げ（seeking・いまのまま）。
"""

from __future__ import annotations

import pytest

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.event_loop import InformationProcessing, Trigger

pytestmark = pytest.mark.real_window  # 門そのものを確かめる（conftest の窓の開き口を使わない）


def _ip(*, present: float = 1.0, names=("パジュ",)):
    a = MagicMock()
    a.config.agent_names = list(names)
    a._pmm.presence_status = MagicMock(return_value=[])
    a._oif.write = AsyncMock(return_value="obs-1")
    a._observation_perspective = MagicMock(return_value={})
    a._conversation_perspective = MagicMock(return_value={})
    a._social_presence_permission = MagicMock(return_value=present)
    a._nudge_seeking = AsyncMock()
    a._timer_tool.frame = MagicMock(return_value="")
    a._dif.ringing = False
    ip = InformationProcessing(a)
    ip._load_silence = lambda: None  # 黙ってはいない
    return ip, a


def _heard(ip, text, source="voice"):
    async def go():
        fut = asyncio.get_running_loop().create_future()
        swallowed = await ip._swallow_if_unheard(
            Trigger(kind="会話入力", query=text, future=fut, source=source)
        )
        return swallowed

    return not asyncio.run(asyncio.wait_for(go(), timeout=2.0))


def test_a_voice_without_the_name_is_dropped_and_not_recorded():
    ip, a = _ip()
    assert _heard(ip, "ごちそうさまでした") is False
    a._oif.write.assert_not_awaited()  # 記録しない
    assert ip._muted == []  # 溜めもしない


def test_the_name_opens_the_window_and_the_next_words_are_heard():
    ip, _ = _ip()
    assert _heard(ip, "パジュ、おはよう") is True
    assert _heard(ip, "明日の天気は？") is True  # 窓の中は名前が要らない


def test_the_window_closes_after_thirty_seconds(monkeypatch):
    ip, _ = _ip()
    clock = [1000.0]
    monkeypatch.setattr("familiar_agent.loop.event_loop.time.monotonic", lambda: clock[0])
    assert _heard(ip, "パジュ") is True
    clock[0] += 29.0
    assert _heard(ip, "ねえ") is True  # 窓の中・延びる
    clock[0] += 31.0
    assert _heard(ip, "聞こえる？") is False  # 延びた 30 秒も過ぎた（出-au）


def test_typing_needs_the_name_like_the_voice():
    """出-au 段 1-2：キーボードも声と同じ。名前で窓を開け、窓の中なら名前は要らない。"""
    ip, _ = _ip()
    assert _heard(ip, "30分だまってて", source="keyboard") is False
    assert _heard(ip, "パジュ、30分だまってて", source="keyboard") is True
    assert _heard(ip, "やっぱりいいや") is True  # 窓の中は声も受ける


def test_nobody_on_camera_does_not_stop_the_window():
    """名前で呼ばれたこと・打たれたことが居る証拠になる（本人の決定）。"""
    ip, a = _ip(present=0.0)
    assert _heard(ip, "パジュ、聞こえる？") is True
    a._nudge_seeking.assert_not_awaited()


def test_a_dropped_voice_with_nobody_visible_still_nudges_seeking():
    ip, a = _ip(present=0.0)
    assert _heard(ip, "こんにちは") is False
    a._nudge_seeking.assert_awaited_once()


def test_without_names_nothing_is_heard():
    """名前が設定されていなければ、声もキーボードも何も受けない（出-au でキーボードも声と同じにした）。"""
    ip, _ = _ip(names=())
    assert _heard(ip, "パジュ、聞こえる？") is False
    assert _heard(ip, "聞こえる？", source="keyboard") is False


def test_a_timer_control_word_needs_the_name():
    """出-au 段 1-2：タイマーの操作にも名前が要る（「パジュ、止めて」で止まる）。"""
    ip, a = _ip()
    a._timer_tool.frame = MagicMock(return_value="[タイマー]\n- id=1 パスタ 鳴っている")
    assert _heard(ip, "止めて") is False
    assert _heard(ip, "パジュ、止めて") is True


def test_a_control_word_without_a_timer_is_just_talk():
    """タイマーが無ければ「止めて」は操作の言葉ではない。名前が無ければ捨てる。"""
    ip, _ = _ip()
    assert _heard(ip, "止めて") is False
