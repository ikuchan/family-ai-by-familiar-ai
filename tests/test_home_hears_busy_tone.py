"""端-b 段 1-は：窓が端末のものなのに家で話しかけられたら、家で短い話し中の音を鳴らす（課題9 §7.4「家での断り方」）。

- 音は作り置き（`sounds/react_busy.wav`）で、主LLM も声の合成も通さない。門は鳴り終わりを待たない。
- 鳴らすのは、窓が空いていれば窓を開けた家の言葉（名前で起こせる声・キーボード）だけ。名前の無い声・1 字違いの名前の声・
  家が黙っているあいだは鳴らさない（仮）。
- 端末の求めを考えているあいだ（耳は聞かない）も鳴らす。口の門を、考えているあいだの門より前に置く。
- 列で待つあいだに端末が窓を取って断った家の言葉にも鳴らす（仮）。端末の言葉を断っても家では鳴らさない。

音は鳴らさない（`DIF` は偽物）。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.loop.event_loop import InformationProcessing, Trigger

pytestmark = pytest.mark.real_window

TAB = "tablet-1"


def _ip(*, silenced: bool = False):
    a = MagicMock()
    a.config.agent_names = ["パジュ"]
    a._pmm.presence_status = MagicMock(return_value=[])
    a._oif.write = AsyncMock(return_value="obs-1")
    a._observation_perspective = MagicMock(return_value={})
    a._conversation_perspective = MagicMock(return_value={})
    a._occupancy = MagicMock(return_value=1.0)
    a._someone_here = MagicMock(return_value=True)
    a._nudge_drive = AsyncMock()
    a._timer_tool.frame = MagicMock(return_value="")
    a._dif.ringing = False
    ip = InformationProcessing(a)
    ip._dif = MagicMock()
    if silenced:
        from familiar_agent.silence_state import SilenceRequest

        ip._paju().silence = SilenceRequest(person="ママ", until=time.time() + 600)
    return ip


def _push(ip, text, *, source="voice", device=""):
    async def go():
        fut = asyncio.get_running_loop().create_future()
        t = Trigger(kind="会話入力", query=text, future=fut, source=source, device=device)
        return not await ip._swallow_if_unheard(t)

    return asyncio.run(asyncio.wait_for(go(), timeout=2.0))


def _device_owns(ip) -> None:
    assert _push(ip, "明日の天気は？", source="device", device=TAB)


def test_a_named_home_voice_hears_the_busy_tone_while_a_device_owns_the_window():
    ip = _ip()
    _device_owns(ip)
    assert _push(ip, "パジュ、おはよう") is False
    ip._dif.play_busy.assert_called_once()


def test_a_typed_home_word_hears_the_busy_tone():
    ip = _ip()
    _device_owns(ip)
    assert _push(ip, "おはよう", source="keyboard") is False
    ip._dif.play_busy.assert_called_once()


def test_no_busy_tone_for_home_talk_that_was_not_to_paju():
    """**反証の側**：名前の無い声（家族どうし・テレビ）と 1 字違いの名前の声では鳴らさない。"""
    ip = _ip()
    _device_owns(ip)
    assert _push(ip, "おはよう") is False
    assert _push(ip, "パジョ、おはよう") is False
    assert _push(ip, "パジュパジュ") is False  # 名前だけの繰り返しは窓を開けない
    ip._dif.play_busy.assert_not_called()


def test_no_busy_tone_while_home_is_silenced():
    ip = _ip(silenced=True)
    _device_owns(ip)
    assert _push(ip, "パジュ、おはよう") is False
    ip._dif.play_busy.assert_not_called()


def test_the_busy_tone_rings_while_paju_thinks_for_the_device():
    """端末の求めのあいだ耳は聞かないにいるが、家の名前の声には話し中の音を返す（黙って捨てない）。"""
    ip = _ip()
    _device_owns(ip)
    ip._paju().is_deaf = lambda: True
    assert _push(ip, "パジュ、おはよう") is False
    ip._dif.play_busy.assert_called_once()


def test_no_busy_tone_at_home_when_a_device_word_is_refused():
    ip = _ip()
    assert _push(ip, "パジュ、おはよう")
    ip._dif.play_busy.reset_mock()
    assert _push(ip, "明日の天気は？", source="device", device=TAB) is False
    ip._dif.play_busy.assert_not_called()


def test_no_busy_tone_while_home_owns_the_window():
    ip = _ip()
    assert _push(ip, "パジュ、おはよう")
    assert _push(ip, "それでね")
    ip._dif.play_busy.assert_not_called()


def test_a_waited_home_word_refused_at_take_time_hears_the_busy_tone():
    ip = _ip()
    _device_owns(ip)
    home = Trigger(kind="会話入力", query="あと", device="")
    assert ip._owned_by_another(home)
    ip._dif.play_busy.assert_called_once()
    ip._dif.play_busy.reset_mock()
    other = Trigger(kind="会話入力", query="傘は？", device="phone-2")
    assert ip._owned_by_another(other)
    ip._dif.play_busy.assert_not_called()


def test_the_busy_tone_is_a_made_sound_file():
    from familiar_agent.core import reaction_cue as rc

    assert rc.BUSY_WAV.exists()
    assert rc.BUSY_WAV in rc.sound_files()


def test_no_busy_tone_while_paju_speaks_at_home():
    """**反証の側**：家の声が話しているあいだ（端末の窓のあいだの家の情動の声）の聞こえ返しには鳴らさない。"""
    ip = _ip()
    _device_owns(ip)
    ip._paju().voice_state = lambda: "speaking"
    assert _push(ip, "パジュ、おはよう") is False
    ip._dif.play_busy.assert_not_called()


def test_the_dif_does_not_stack_busy_tones(monkeypatch):
    """続けて呼ばれても、鳴っているあいだは重ねない。待たない（同期の口）。"""
    from familiar_agent.io import dif as dif_mod

    async def go():
        played: list = []
        gate = asyncio.Event()

        async def fake_play(path, gain, quiet=False):
            played.append(path)
            await gate.wait()
            return True

        monkeypatch.setattr(dif_mod, "_play_wav", fake_play)
        d = dif_mod.DIF.__new__(dif_mod.DIF)
        d.play_busy()
        d.play_busy()
        await asyncio.sleep(0.01)
        assert len(played) == 1
        gate.set()
        await asyncio.sleep(0.01)
        gate.clear()
        d.play_busy()
        await asyncio.sleep(0.01)
        assert len(played) == 2
        gate.set()
        await d._busy_task

    asyncio.run(go())


def test_no_busy_tone_at_take_time_while_paju_speaks_or_home_is_silenced():
    """**反証の側**：列から取り出すときに断った家の言葉も、家の声が話しているあいだと黙っているあいだは鳴らさない。"""
    ip = _ip()
    _device_owns(ip)
    ip._paju().voice_state = lambda: "speaking"
    assert ip._owned_by_another(Trigger(kind="会話入力", query="あと", device=""))
    ip._dif.play_busy.assert_not_called()

    ip = _ip(silenced=True)
    _device_owns(ip)
    assert ip._owned_by_another(Trigger(kind="会話入力", query="あと", device=""))
    ip._dif.play_busy.assert_not_called()
