"""声の区切りから特徴を取り出し、書き起こしの印に載せて運ぶ（知-ae 段 3・2026-10-02）。

書き起こしは音を文字にしたら捨てていたので、照合に渡る音が無かった。区切りごとに、書き起こしと並べて声の特徴
（ECAPA）を取り出し、`VoiceText.voice` に載せて待ち行列へ積む。

- 照らすのは 1.5 秒以上の区切りだけ。持ち越して単独で配った短い断片には載せない（短いと特徴が揺れる）。
- 取り出しに失敗しても、書き起こしは配る（声が分からないだけ）。
- 中継（`_committed_relay`）が印を付け直しても、特徴は落とさない。
- 特徴はファイルを介さず PCM から取る。モデルは CPU に置く（Whisper と GPU を取り合わない）。
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from familiar_agent.core.wake_window import VoiceText, voice_of
from tests.test_local_stt import _LOUD_FRAME, _RATE, FRAME_SAMPLES, _engine, _feed

VEC = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
LONG = int(1.6 * _RATE / FRAME_SAMPLES)


def test_a_long_segment_carries_its_voice():
    engine = _engine(vad_says=["start"] + [None] * (LONG - 1) + ["end"])
    engine._embed = MagicMock(return_value=VEC)
    _feed(engine, [_LOUD_FRAME] * (LONG + 1))
    got = engine.on_committed.get_nowait()
    assert got == "おはよう" and isinstance(got, VoiceText)
    assert np.allclose(voice_of(got), VEC)
    assert engine._embed.call_args.args[0] == engine._transcribe.call_args.args[0]  # 同じ区切りの音


def test_a_short_fragment_given_up_alone_carries_no_voice():
    engine = _engine(vad_says=["start", None, "end"] + [None] * 5)
    engine._embed = MagicMock(return_value=VEC)

    async def run():
        for f in [_LOUD_FRAME] * 3:
            await engine.feed(f)
        engine._held_since = 0.0
        for f in [b"\x00\x00" * FRAME_SAMPLES] * 2:
            await engine.feed(f)

    asyncio.run(run())
    got = engine.on_committed.get_nowait()
    assert got == "おはよう" and voice_of(got) is None
    engine._embed.assert_not_called()


def test_a_failed_extraction_still_delivers_the_words():
    engine = _engine(vad_says=["start"] + [None] * (LONG - 1) + ["end"])
    engine._embed = MagicMock(side_effect=RuntimeError("model"))
    _feed(engine, [_LOUD_FRAME] * (LONG + 1))
    got = engine.on_committed.get_nowait()
    assert got == "おはよう" and voice_of(got) is None


def test_plain_text_has_no_voice():
    assert voice_of("おはよう") is None
    assert voice_of(VoiceText("おはよう")) is None


@pytest.mark.asyncio
async def test_the_relay_keeps_the_voice():
    from familiar_agent.realtime_stt_session import RealtimeSttSession

    session = RealtimeSttSession("dummy")
    committed_q: asyncio.Queue = asyncio.Queue()
    input_q: asyncio.Queue = asyncio.Queue()
    session._incoming_committed = committed_q
    session._committed_queue = input_q
    session.mic_gate = lambda: ""
    task = asyncio.create_task(session._committed_relay())
    await committed_q.put(VoiceText("パジュ、パパだよ", voice=VEC))
    await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    got = input_q.get_nowait()
    assert got == "パジュ、パパだよ" and np.allclose(voice_of(got), VEC)


def test_the_feature_is_taken_from_pcm_on_the_cpu():
    from familiar_agent.recognition import voice as voice_mod

    class _Model:
        def encode_batch(self, signal):
            import torch

            assert signal.dtype == torch.float32 and signal.shape == (1, 4)
            assert float(signal.abs().max()) <= 1.0
            return torch.ones(1, 1, 3)

    pcm = np.asarray([0, 16384, -16384, 32767], dtype="<i2").tobytes()
    with patch.object(voice_mod, "_get_model", return_value=_Model()):
        got = voice_mod.embed_pcm(pcm)
    assert got is not None and got.shape == (3,)
    with patch.object(voice_mod, "_get_model", return_value=None):
        assert voice_mod.embed_pcm(pcm) is None
    assert voice_mod.embed_pcm(b"") is None


def test_the_model_is_loaded_on_the_cpu():
    from familiar_agent.recognition import voice as voice_mod

    seen = {}

    class _EC:
        @staticmethod
        def from_hparams(**kw):
            seen.update(kw)
            return object()

    fake = MagicMock()
    fake.EncoderClassifier = _EC
    with (
        patch.object(voice_mod, "_MODEL", None),
        patch.dict("sys.modules", {"speechbrain.inference.speaker": fake}),
    ):
        voice_mod._get_model()
    assert seen["run_opts"] == {"device": "cpu"}


# ── 書き起こしの確かさ（名前で起きる基準・2026-10-05）─────────────────────────


def test_the_measurements_ride_the_transcript():
    """区切りの無音らしさ（いちばん大きい値）と確かさ（いちばん低い値）を、入力の印に載せて運ぶ。"""
    from familiar_agent.core.wake_window import measures_of

    engine = _engine(vad_says=["start"] + [None] * (LONG - 1) + ["end"])

    def transcribe(audio):
        engine._last_measures = (0.150, -0.723)
        return "パジュー"

    engine._transcribe = MagicMock(side_effect=transcribe)
    _feed(engine, [_LOUD_FRAME] * (LONG + 1))
    got = engine.on_committed.get_nowait()
    assert measures_of(got) == (0.150, -0.723)


def test_plain_text_has_no_measurements():
    from familiar_agent.core.wake_window import measures_of

    assert measures_of("パジュ") == (None, None)
    assert measures_of(VoiceText("パジュ")) == (None, None)


@pytest.mark.asyncio
async def test_the_relay_keeps_the_measurements():
    from familiar_agent.core.wake_window import measures_of
    from familiar_agent.realtime_stt_session import RealtimeSttSession

    session = RealtimeSttSession("dummy")
    committed_q: asyncio.Queue = asyncio.Queue()
    input_q: asyncio.Queue = asyncio.Queue()
    session._incoming_committed = committed_q
    session._committed_queue = input_q
    session.mic_gate = lambda: ""
    task = asyncio.create_task(session._committed_relay())
    await committed_q.put(VoiceText("パジュー", no_speech=0.15, logprob=-0.72))
    await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert measures_of(input_q.get_nowait()) == (0.15, -0.72)


def test_run_hands_the_measurements_to_the_loop():
    from familiar_agent.agent import EmbodiedAgent as Agent
    from tests.test_input_commands_before_loop_branch import _agent as _run_agent

    a = _run_agent()
    asyncio.run(Agent.run(a, VoiceText("パジュー", no_speech=0.15, logprob=-0.72)))
    kw = a._info_processing.push_utterance.await_args.kwargs
    assert (kw["no_speech"], kw["logprob"]) == (0.15, -0.72)
