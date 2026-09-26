"""入力に「届いた時刻」と「出どころ」を付けて運ぶ（出-au 段 1-1・2026-09-26・`設計方針_判定の段` §2 原則 3）。

窓の判定は、駆動体が取り出した時刻ではなく**届いた時刻**で行う。画面は前の `run()` が返るまで次の入力を
取らないので、取り出した時刻で見ると、窓の中で言った声でも窓が切れた後として捨てることがある。
時刻は積むところ（声の書き起こし・キーボード 3 か所）で付け、`agent.run` の先頭（名前の札を外す前）で読む。
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect

import pytest

from familiar_agent.core.wake_window import KeyText, VoiceText, arrived_at, source_of
from familiar_agent.realtime_stt_session import RealtimeSttSession


def test_marked_text_carries_its_arrival_and_source():
    v = VoiceText("パジュ、おはよう", at=12.5)
    k = KeyText("パジュ、おはよう", at=3.0)
    assert v == k == "パジュ、おはよう" and isinstance(v, str)
    assert (source_of(v), arrived_at(v)) == ("voice", 12.5)
    assert (source_of(k), arrived_at(k)) == ("keyboard", 3.0)


def test_plain_text_arrives_now():
    assert arrived_at("おはよう", now=99.0) == 99.0
    assert source_of("おはよう") == "keyboard"


def test_marking_stamps_the_clock_when_no_time_is_given(monkeypatch):
    monkeypatch.setattr("familiar_agent.core.wake_window.time.monotonic", lambda: 42.0)
    assert arrived_at(VoiceText("x")) == 42.0


@pytest.mark.asyncio
async def test_the_stt_relay_stamps_what_it_heard():
    # 時計は差し替えない（asyncio も同じ `time.monotonic` を使うので、止めると sleep が終わらない）。
    import time

    before = time.monotonic()
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
    assert source_of(got) == "voice" and before <= arrived_at(got) <= time.monotonic()


@pytest.mark.parametrize(
    "module, marker",
    [
        ("familiar_agent.gui", "self._input_queue.put_nowait(KeyText(text))"),
        ("familiar_agent.tui", "await self._input_queue.put(KeyText(text))"),
        ("familiar_agent.tui", "await self._input_queue.put(VoiceText(text))"),
        ("familiar_agent.main", "await input_queue.put(KeyText(line.strip()))"),
    ],
)
def test_every_keyboard_and_voice_door_stamps_its_input(module, marker):
    """積む口は 4 つ（画面・TUI の打鍵と録音・CUI）。どれかが素の文字列を積むと、その入力は取り出した時刻で判定される。"""
    import importlib

    pytest.importorskip("PySide6") if module.endswith("gui") else None
    src = inspect.getsource(importlib.import_module(module))
    assert marker in src


def test_run_reads_arrival_before_stripping_the_speaker_tag():
    """`[パパ] …` の札を外すと印の無い文字列になる。時刻は外す前に読む。"""
    from familiar_agent.agent import EmbodiedAgent as Agent
    from tests.test_input_commands_before_loop_branch import _agent

    a = _agent()
    asyncio.run(Agent.run(a, VoiceText("[パパ] パジュ、おはよう", at=5.0)))
    kw = a._info_processing.push_utterance.await_args.kwargs
    assert kw["arrived"] == 5.0 and kw["source"] == "voice"


def test_the_trigger_carries_its_arrival():
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    async def scenario():
        ip = InformationProcessing(_agent(stream_returns=[]))
        ip._ensure_driver = lambda: None
        task = asyncio.ensure_future(ip.push_utterance("おはよう", source="voice", arrived=8.0))
        await asyncio.sleep(0.01)
        trig = ip._triggers.get_nowait()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await ip.close()
        return trig

    assert asyncio.run(scenario()).arrived == 8.0
