"""環-aw 第 5 段 項目 2：dif が口の声の事象を送る（`設計方針_パジュの状態` §3.5）。

声（`speak`、`say_recorded`）は、状態図の声の鍵を取ってから「話し始めた」（`voice_began`）を送り、終わりに
「話し終えた」（`voice_ended`）を送る。声の鍵は状態図に 1 つで、ループの dif と自律機構の dif が同じ鍵を使う。
考え中の音（`start_thinking`）と合図（`play_cue`）も、始まりと終わりを送る。状態図が無ければ、事象を送らずに鳴らす。
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from familiar_agent.core import reaction_cue as rc
from familiar_agent.io import dif as dif_mod
from familiar_agent.loop.paju_state import PajuState


class _IP:
    """状態図を 1 つ持つ偽のループ（`_paju()` だけ）。"""

    def __init__(self) -> None:
        self.sm = PajuState()
        self.sent: list[tuple[str, dict]] = []
        fire = self.sm.fire

        async def _spy(event: str, **data: object) -> None:
            self.sent.append((event, data))
            await fire(event, **data)

        self.sm.fire = _spy  # type: ignore[method-assign]

    def _paju(self) -> PajuState:
        return self.sm

    def events(self) -> list[str]:
        return [e for e, _ in self.sent]


class _TTS:
    """`say` を鳴らす偽の担い手。鳴っているあいだの声の状態を残し、`gate` が立つまで鳴り終えない。"""

    def __init__(self, ip: _IP, gate: "asyncio.Event | None" = None) -> None:
        self._ip = ip
        self.gate = gate
        self.seen: list[str] = []
        self.playing = 0
        self.most = 0

    async def call(self, name: str, payload: dict):
        self.playing += 1
        self.most = max(self.most, self.playing)
        self.seen.append(self._ip.sm.voice_state())
        try:
            if self.gate is not None:
                await self.gate.wait()
            await asyncio.sleep(0)
        finally:
            self.playing -= 1
        return f"Said: {payload['text']}", None


def _quiet_wav(monkeypatch, hold: "dict[str, asyncio.Event] | None" = None, seen=None, ip=None):
    """`_play_wav` を偽にする。`hold` に名前のある wav は、その印が立つまで鳴り終えない。"""

    async def _play(path, gain, *, quiet=False):
        if seen is not None and ip is not None:
            seen.append(ip.sm.voice_state())
        ev = (hold or {}).get(path.name)
        if ev is not None:
            await ev.wait()
        await asyncio.sleep(0)
        return True

    monkeypatch.setattr(dif_mod, "_play_wav", _play)


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


# ── 声（speak・say_recorded）──────────────────────────────────────────────


def test_a_voice_sends_began_and_ended_around_the_playing():
    async def go():
        ip = _IP()
        await ip.sm.start()
        tts = _TTS(ip)
        await dif_mod.DIF(tts=tts, ip=ip).speak("こんにちは")
        assert tts.seen == ["speaking"]
        assert ip.sent == [("voice_began", {}), ("voice_ended", {"played": True})]
        assert ip.sm.voice_state() == "no_voice"

    asyncio.run(go())


def test_a_recording_is_a_voice_too(monkeypatch):
    async def go():
        ip = _IP()
        await ip.sm.start()
        seen: list[str] = []
        _quiet_wav(monkeypatch, seen=seen, ip=ip)
        d = dif_mod.DIF(ip=ip)
        assert await d.say_recorded(rc.TIMEOUT_RECORDING, gain=1.0)
        assert await d.say_credit("gemini", gain=0.25)  # 合図を待たず、音楽も絞らない録音も声
        assert seen == ["speaking", "speaking"]
        assert ip.events() == ["voice_began", "voice_ended"] * 2
        assert ip.sent[1] == ("voice_ended", {"played": True})

    asyncio.run(go())


def test_two_voices_take_turns_even_from_two_difs(monkeypatch):
    """ループの dif と自律機構の dif は、状態図の 1 つの声の鍵で 1 本ずつ通る。"""

    async def go():
        ip = _IP()
        await ip.sm.start()
        gate = asyncio.Event()
        tts = _TTS(ip, gate)
        hold = {"credit_gemini.wav": asyncio.Event()}
        _quiet_wav(monkeypatch, hold)
        loop_dif = dif_mod.DIF(tts=tts, ip=ip)
        tonic_dif = dif_mod.DIF(ip=ip)
        a = asyncio.ensure_future(loop_dif.speak("一つ目"))
        await _settle()
        b = asyncio.ensure_future(tonic_dif.say_credit("gemini", gain=0.25))
        await _settle()
        assert ip.events() == ["voice_began"]  # 2 つ目は鍵を待っている
        gate.set()
        await a
        await _settle()
        hold["credit_gemini.wav"].set()
        await b
        assert ip.events() == ["voice_began", "voice_ended"] * 2
        assert tts.most == 1
        assert ip.sm.voice_state() == "no_voice"

    asyncio.run(go())


def test_a_voice_cancelled_while_waiting_sends_nothing():
    async def go():
        ip = _IP()
        await ip.sm.start()
        gate = asyncio.Event()
        d = dif_mod.DIF(tts=_TTS(ip, gate), ip=ip)
        a = asyncio.ensure_future(d.speak("一つ目"))
        await _settle()
        b = asyncio.ensure_future(d.speak("二つ目"))
        await _settle()
        b.cancel()
        await _settle()
        gate.set()
        await a
        assert ip.events() == ["voice_began", "voice_ended"]

    asyncio.run(go())


def test_a_voice_cancelled_while_playing_still_ends():
    async def go():
        ip = _IP()
        await ip.sm.start()
        d = dif_mod.DIF(tts=_TTS(ip, asyncio.Event()), ip=ip)
        a = asyncio.ensure_future(d.speak("途中で止める"))
        await _settle()
        assert ip.sm.voice_state() == "speaking"
        a.cancel()
        await _settle()
        assert ip.events() == ["voice_began", "voice_ended"]
        assert ip.sent[-1] == ("voice_ended", {"played": False})
        assert ip.sm.voice_state() == "no_voice"

    asyncio.run(go())


def test_a_voice_that_did_not_play_ends_with_played_false():
    async def go():
        ip = _IP()
        await ip.sm.start()
        tts = MagicMock()

        async def _fail(name, payload):
            return "402 Payment Required", None

        tts.call = _fail
        await dif_mod.DIF(tts=tts, ip=ip).speak("出ない声")
        assert ip.sent[-1] == ("voice_ended", {"played": False})

    asyncio.run(go())


# ── 考え中の音と合図 ──────────────────────────────────────────────────────


def test_the_thinking_cue_sends_began_and_ended(monkeypatch):
    async def go():
        ip = _IP()
        await ip.sm.start()
        _quiet_wav(monkeypatch, {rc.THINKING_WAV.name: asyncio.Event()})
        thinking = dif_mod.DIF(ip=ip).start_thinking()
        await _settle()
        assert ip.sm.voice_state() == "thinking_cue"
        thinking.stop()
        await _settle()
        assert ip.events() == ["thinking_cue_began", "thinking_cue_ended"]
        assert ip.sm.voice_state() == "no_voice"

    asyncio.run(go())


def test_a_cue_after_the_thinking_cue_stays_a_cue(monkeypatch):
    """考え中の音を止めてすぐ合図を鳴らすと、遅れて届いた考え中の音の終わりは捨てられ、合図のまま。"""

    async def go():
        ip = _IP()
        await ip.sm.start()
        hold = {rc.THINKING_WAV.name: asyncio.Event(), rc.ACK_WAV.name: asyncio.Event()}
        _quiet_wav(monkeypatch, hold)
        d = dif_mod.DIF(ip=ip)
        thinking = d.start_thinking()
        await _settle()
        thinking.stop()
        d.play_cue(rc.ACK)
        await _settle()
        assert ip.sm.voice_state() == "cue"
        hold[rc.ACK_WAV.name].set()
        await _settle()
        assert ip.sm.voice_state() == "no_voice"

    asyncio.run(go())


def test_only_the_current_cue_ends_the_cue(monkeypatch):
    """**反証の側**：合図が 2 つ続いたとき、前の合図の終わりで後の合図を終わらせない。"""

    async def go():
        ip = _IP()
        await ip.sm.start()
        hold = {rc.ACK_WAV.name: asyncio.Event(), rc.SILENT_WAV.name: asyncio.Event()}
        _quiet_wav(monkeypatch, hold)
        d = dif_mod.DIF(ip=ip)
        d.play_cue(rc.ACK)
        await _settle()
        d.play_cue(rc.SILENT)
        await _settle()
        hold[rc.ACK_WAV.name].set()  # 前の合図が終わる
        await _settle()
        assert ip.sm.voice_state() == "cue"
        hold[rc.SILENT_WAV.name].set()
        await _settle()
        assert ip.sm.voice_state() == "no_voice"
        assert ip.events() == ["cue_began", "cue_began", "cue_ended"]

    asyncio.run(go())


def test_a_cue_that_does_not_sound_sends_nothing(monkeypatch):
    """**反証の側**：鳴らす音の無い合図（なし）は事象を送らず、声はなしのまま（待ちの電子音を休ませない）。"""

    async def go():
        ip = _IP()
        await ip.sm.start()
        _quiet_wav(monkeypatch)
        assert dif_mod.DIF(ip=ip).play_cue(rc.NONE) is None
        await _settle()
        assert ip.sent == [] and ip.sm.voice_state() == "no_voice"

    asyncio.run(go())


def test_speech_waits_for_the_cue_then_takes_over(monkeypatch):
    async def go():
        ip = _IP()
        await ip.sm.start()
        hold = {rc.ACK_WAV.name: asyncio.Event()}
        _quiet_wav(monkeypatch, hold)
        tts = _TTS(ip)
        d = dif_mod.DIF(tts=tts, ip=ip)
        d.play_cue(rc.ACK)
        a = asyncio.ensure_future(d.speak("はい"))
        await _settle()
        assert ip.sm.voice_state() == "cue" and tts.seen == []
        hold[rc.ACK_WAV.name].set()
        await a
        assert tts.seen == ["speaking"]
        assert ip.sm.voice_state() == "no_voice"

    asyncio.run(go())


# ── 状態図が無いとき ──────────────────────────────────────────────────────


def test_without_a_chart_voices_play_as_before(monkeypatch):
    """**反証の側**：`ip` が無い、または状態図を持たない偽物（MagicMock）なら、事象を送らずに鳴らす。"""

    async def go():
        played: list[str] = []

        async def _play(path, gain, *, quiet=False):
            played.append(path.name)
            return True

        monkeypatch.setattr(dif_mod, "_play_wav", _play)
        for ip in (None, MagicMock()):
            tts = MagicMock()

            async def _say(name, payload):
                played.append(payload["text"])
                return "Said: x", None

            tts.call = _say
            d = dif_mod.DIF(tts=tts, ip=ip)
            await d.speak("声")
            assert await d.say_recorded(rc.TIMEOUT_RECORDING, gain=1.0)
            task = d.play_cue(rc.ACK)
            assert task is not None
            await task
        assert played.count("声") == 2 and played.count(rc.ACK_WAV.name) == 2

    asyncio.run(go())


# ── 黙っているあいだ（項目 6）──────────────────────────────────────────────


async def _silenced(ip: "_IP") -> None:
    import time

    from familiar_agent.silence_state import SilenceRequest

    await ip.sm.start()
    await ip.sm.hush_with(SilenceRequest(person="パパ", until=time.time() + 60, reason=""))
    ip.sent.clear()


def test_a_voice_outside_a_request_is_held_while_silenced(monkeypatch):
    """黙っているあいだ、求めの外の声（印 `through_silence` が無い）は鳴らさず None を返し、事象も送らない。"""

    async def go():
        ip = _IP()
        await _silenced(ip)
        played: list[str] = []
        _quiet_wav(monkeypatch, seen=played, ip=ip)
        tts = _TTS(ip)
        d = dif_mod.DIF(tts=tts, ip=ip)
        assert await d.speak("申し出") is None
        assert await d.say_recorded(rc.TIMEOUT_RECORDING, gain=1.0) is None
        assert await d.say_credit("gemini", gain=0.25) is None
        assert tts.seen == [] and played == []
        assert ip.events() == [] and ip.sm.voice_state() == "no_voice"

    asyncio.run(go())


def test_a_voice_inside_a_request_passes_through_silence(monkeypatch):
    """黙っているあいだでも、求めの中の声（印 `through_silence`）は鳴らす（タイマーの返事・鳴った知らせ）。"""

    async def go():
        ip = _IP()
        await _silenced(ip)
        _quiet_wav(monkeypatch)
        tts = _TTS(ip)
        d = dif_mod.DIF(tts=tts, ip=ip)
        assert await d.speak("タイマーを3分にセットしました", through_silence=True) is True
        assert await d.say_recorded(rc.TIMEOUT_RECORDING, gain=1.0, through_silence=True)
        assert tts.seen == ["speaking"]
        assert ip.events() == ["voice_began", "voice_ended"] * 2

    asyncio.run(go())


def test_a_voice_is_not_held_when_not_silenced():
    """**反証の側**：黙っていなければ、印が無くても鳴らす。鳴らなかった声は None ではなく偽。"""

    async def go():
        ip = _IP()
        await ip.sm.start()
        assert await dif_mod.DIF(tts=_TTS(ip), ip=ip).speak("こんにちは") is True
        tts = MagicMock()

        async def _fail(name, payload):
            return "402 Payment Required", None

        tts.call = _fail
        assert await dif_mod.DIF(tts=tts, ip=ip).speak("出ない声") is False

    asyncio.run(go())


def test_an_expired_silence_does_not_hold():
    """**反証の側**：黙っているかは鳴らす時刻で決める。期限を過ぎた依頼では止めない。"""

    async def go():
        import time

        from familiar_agent.silence_state import SilenceRequest

        ip = _IP()
        await ip.sm.start()
        await ip.sm.hush_with(SilenceRequest(person="パパ", until=time.time() + 0.05, reason=""))
        await asyncio.sleep(0.1)
        assert await dif_mod.DIF(tts=_TTS(ip), ip=ip).speak("こんにちは") is True

    asyncio.run(go())


def test_a_held_voice_does_not_wait_for_another_voice(monkeypatch):
    """黙っていれば、ほかの声が鍵を持っていても待たずに None を返す（自律機構の輪を止めない・見直しの指摘）。"""

    async def go():
        ip = _IP()
        await _silenced(ip)
        gate = asyncio.Event()
        _quiet_wav(monkeypatch)
        loop_dif = dif_mod.DIF(tts=_TTS(ip, gate), ip=ip)
        a = asyncio.ensure_future(loop_dif.speak("長い答え", through_silence=True))
        await _settle()
        assert ip.sm.voice_state() == "speaking"
        held = await asyncio.wait_for(dif_mod.DIF(ip=ip).say_credit("gemini", gain=0.25), 1.0)
        assert held is None
        gate.set()
        await a

    asyncio.run(go())


def test_a_broken_chart_does_not_silence_paju(monkeypatch, caplog):
    """**反証の側**：状態図が例外を投げても、声、合図、機械音 A は鳴らし、警告を残す（状態図が落ちても会話は止めない）。"""

    async def go():
        ip = _IP()
        await ip.sm.start()

        async def broken(event, **data):
            raise RuntimeError("状態図の動作が落ちた")

        ip.sm.fire = broken  # type: ignore[method-assign]
        played: list[str] = []

        async def _play(path, gain, *, quiet=False):
            played.append(path.name)
            return path.name != rc.THINKING_WAV.name  # 機械音 A は 1 回で止める

        monkeypatch.setattr(dif_mod, "_play_wav", _play)
        tts = _TTS(ip)
        d = dif_mod.DIF(tts=tts, ip=ip)
        assert await d.speak("こんにちは") is True
        task = d.play_cue(rc.ACK)
        assert task is not None
        await task
        d.start_thinking()
        await _settle()
        assert len(tts.seen) == 1
        assert rc.ACK_WAV.name in played and rc.THINKING_WAV.name in played

    import logging

    with caplog.at_level(logging.WARNING, logger="familiar_agent.io.dif"):
        asyncio.run(go())
    assert sum("状態図へ" in r.getMessage() for r in caplog.records) >= 3
