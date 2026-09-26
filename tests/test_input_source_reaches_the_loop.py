"""入力の出どころ（声かキーボードか）をループまで運ぶ（出-as 段 2・2026-09-26）。

ウェイクワードの窓は声にだけ掛け、キーボードはいつでも受ける（`設計方針_話していいかの決まり` v0.1 §2.3）。
ところが出どころは GUI がログに書くだけで、待ち行列には文字だけが積まれ、ループまで届いていなかった。

声の書き起こしを積むところで**印つきの文字列**（`VoiceText`）にし、`agent.run` の入口で印を見て決める。
画面（GUI・TUI・CUI）は変えない。印が無ければキーボード——`.env.quiet`（マイクを聞かない）の入力もこれ。
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from familiar_agent.core.wake_window import VoiceText, source_of
from familiar_agent.realtime_stt_session import RealtimeSttSession


def test_voice_text_is_still_the_text():
    t = VoiceText("パジュ、おはよう")
    assert isinstance(t, str) and t == "パジュ、おはよう"
    assert source_of(t) == "voice"
    assert source_of("パジュ、おはよう") == "keyboard"


@pytest.mark.asyncio
async def test_the_stt_relay_marks_what_it_heard():
    session = RealtimeSttSession("dummy")
    committed_q: asyncio.Queue[str] = asyncio.Queue()
    input_q: asyncio.Queue = asyncio.Queue()
    session._incoming_committed = committed_q
    session._committed_queue = input_q
    session.mic_gate = lambda: ""
    task = asyncio.create_task(session._committed_relay())
    await committed_q.put("パジュ、おはよう")
    await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    got = input_q.get_nowait()
    assert got == "パジュ、おはよう" and source_of(got) == "voice"


def _run(user_input):
    from familiar_agent.agent import EmbodiedAgent as Agent
    from tests.test_input_commands_before_loop_branch import _agent

    a = _agent()
    asyncio.run(Agent.run(a, user_input))
    return a._info_processing.push_utterance.await_args


def test_run_passes_voice_as_voice():
    call = _run(VoiceText("パジュ、おはよう"))
    assert call.kwargs["source"] == "voice"


def test_run_passes_typed_text_as_keyboard():
    call = _run("パジュ、おはよう")
    assert call.kwargs["source"] == "keyboard"


def test_the_trigger_carries_the_source():
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    async def scenario():
        ip = InformationProcessing(_agent(stream_returns=[]))
        ip._ensure_driver = lambda: None  # 駆動体を起こさず、積んだものだけを見る
        task = asyncio.ensure_future(ip.push_utterance("おはよう", source="voice"))
        await asyncio.sleep(0.01)
        trig = ip._triggers.get_nowait()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await ip.close()
        return trig

    trig = asyncio.run(scenario())
    assert trig.kind == "会話入力" and trig.source == "voice"
