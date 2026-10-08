"""声の届いた時刻は、話し始めた時刻（出-bb・2026-10-08・本人の決定ア）。

届いた時刻（`VoiceText.at`）を書き起こしが済んだときに打っていたので、声が長いほど窓が実質短くなった。実機 18:19 の
「週末の天気は」は、話し始めが窓の 1.5 秒後、届いた時刻は 5.8 秒後だった（声 2.9 秒＋無音 1.0 秒＋書き起こし 0.4 秒）。
中継（`realtime_stt_session`）も作り直すときに時刻を打ち直していた。書き起こしが VAD の話し始めを覚えて渡し、中継は
そのまま運ぶ。持ち越した短い断片に続けて話したときは、最初の断片の話し始め（ひと続きの言葉として扱っている）。
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from familiar_agent.core.wake_window import VoiceText, arrived_at
from familiar_agent.realtime_stt_session import RealtimeSttSession
from familiar_agent.tools.local_stt import FRAME_SAMPLES
from tests.test_local_stt import _LOUD_FRAME, _engine

_LONG = int(1.6 * 16000 / FRAME_SAMPLES)  # 1.6 秒（持ち越さない長さ）


_GAP = 0.3  # 話し始めから区切りまでの間（本物の時計で測れる長さ）


def _utterances(engine, *pattern):
    """`pattern` は区切りごとのフレーム数。区切りの話し始めの前後の時刻と、終わった時刻を返す。

    時計は差し替えない（asyncio も同じ `time.monotonic` を使うので、止めると書き起こしの待ちが終わらない）。
    話し始めのフレームの後に `_GAP` 秒あけてから残りを流し、話し始めと書き起こしの済んだ時刻を離す。
    """
    import time

    marks: list = []

    async def run():
        for frames in pattern:
            says = ["start"] + [None] * (frames - 2) + ["end"]
            before = time.monotonic()
            engine._vad_step.side_effect = list(says)
            await engine.feed(_LOUD_FRAME)
            after = time.monotonic()
            await asyncio.sleep(_GAP)
            for _ in says[1:]:
                await engine.feed(_LOUD_FRAME)
            marks.append((before, after, time.monotonic()))

    asyncio.run(run())
    return marks


def test_a_voice_arrives_when_it_started():
    engine = _engine(vad_says=[])
    ((before, after, done),) = _utterances(engine, _LONG)
    got = arrived_at(engine.on_committed.get_nowait())
    assert before <= got <= after  # 書き起こしが済んだ時刻（done）ではない
    assert done - got >= _GAP


def test_a_held_fragment_keeps_the_first_start():
    engine = _engine(vad_says=[])
    (first, _second) = _utterances(engine, 3, _LONG)  # 3 フレームは短いので持ち越す
    got = arrived_at(engine.on_committed.get_nowait())
    assert first[0] <= got <= first[1]


def test_a_new_utterance_after_a_commit_starts_fresh():
    engine = _engine(vad_says=[])
    (_first, second) = _utterances(engine, _LONG, _LONG)
    engine.on_committed.get_nowait()
    got = arrived_at(engine.on_committed.get_nowait())
    assert second[0] <= got <= second[1]


@pytest.mark.asyncio
async def test_the_relay_keeps_when_the_voice_started():
    session = RealtimeSttSession("dummy")
    committed_q: asyncio.Queue = asyncio.Queue()
    input_q: asyncio.Queue = asyncio.Queue()
    session._incoming_committed = committed_q
    session._committed_queue = input_q
    session.mic_gate = lambda: ""
    task = asyncio.create_task(session._committed_relay())
    await committed_q.put(VoiceText("パジュ、週末の天気は", at=12.5, no_speech=0.1, logprob=-0.3))
    await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    got = input_q.get_nowait()
    assert arrived_at(got) == 12.5
