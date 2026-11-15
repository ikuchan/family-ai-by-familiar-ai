"""音楽が鳴っているあいだは、情動が発火しても何もせずに終える（環-ba・2026-11-15・本人の決定）。

鳴っているあいだは名前の無い声を聞かないので、パジュから話しかけても返事を受けられない。発火（溜まった情動が戻る）は
いままでどおり起きるが、QA に積まず、rest の内省へも回さない。rest はイベントループを通らないので、止めるのは T の発火の口。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from familiar_agent.core.drive_dynamics import DriveFiring
from familiar_agent.drive_register import AiDrivers
from familiar_agent.loop.tonic import Tonic

AXES = ("seeking", "rest", "bond", "safety", "esteem")


def _run(axis: str, *, playing: bool):
    ip = MagicMock()
    ip.push_affect = MagicMock()
    agent = SimpleNamespace(_music_state=SimpleNamespace(playing=playing))
    firing = DriveFiring(**{a: a == axis for a in AXES})
    rest = AsyncMock()
    step = AsyncMock(return_value=(firing, AiDrivers()))

    async def scenario():
        with (
            patch("familiar_agent.loop.tonic.step_drives", new=step),
            patch("familiar_agent.loop.tonic.run_rest_pass", new=rest),
        ):
            t = Tonic(ip, period=0.01, agent=agent)
            t.start()
            for _ in range(200):
                if ip.push_affect.call_count or rest.await_count or step.await_count >= 3:
                    break
                await asyncio.sleep(0.005)
            await t.close()

    asyncio.run(scenario())
    assert step.await_count >= 1  # 発火の判定までは回った（何もせずに通ったのではない）
    return ip.push_affect.call_count + rest.await_count


@pytest.mark.parametrize("axis", AXES)
def test_a_firing_does_nothing_while_music_plays(axis, caplog):
    with caplog.at_level("INFO"):
        assert _run(axis, playing=True) == 0
    assert "音楽が鳴っているので何もせずに終える" in caplog.text


@pytest.mark.parametrize("axis", AXES)
def test_a_firing_acts_when_music_is_not_playing(axis):
    """**反証の側**：鳴っていなければ、いままでどおり QA に積む（rest は内省へ回す）。"""
    assert _run(axis, playing=False) > 0


# ── 鳴る前に積まれていた発火も、取り出したときに鳴っていれば何もしない（本人の決定 2） ──────────────


def _affect(playing: bool):
    from familiar_agent.loop.event_loop import InformationProcessing

    from tests.test_event_loop import _agent

    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._agent._music_state = SimpleNamespace(playing=playing)
    ip._write_origin = AsyncMock(return_value=1)  # type: ignore[method-assign]
    ip._iterate = AsyncMock(return_value="")  # type: ignore[method-assign]
    ip._start_beeps = lambda: None  # type: ignore[method-assign]

    async def go():
        await ip._begin_affect("bond", "促し")
        opened = ip._req.request_id
        await ip.close()
        return opened

    return ip, asyncio.run(go())


def test_a_queued_firing_does_nothing_once_music_plays():
    ip, opened = _affect(playing=True)
    assert opened is None
    ip._iterate.assert_not_awaited()


def test_a_queued_firing_opens_when_music_is_not_playing():
    """**反証の側**：鳴っていなければ、いままでどおり求めを開いて考える。"""
    ip, opened = _affect(playing=False)
    assert opened is not None
    ip._iterate.assert_awaited_once()


def test_one_check_decides_whether_music_plays():
    """鳴っているかは `music_state.playing` の 1 か所で決める（耳の入口、発火の止め方、ループの始めがずれない）。"""
    from familiar_agent.core.music_state import playing

    assert playing(SimpleNamespace(_music_state=SimpleNamespace(playing=True)))
    assert not playing(SimpleNamespace(_music_state=SimpleNamespace(playing=False)))
    assert not playing(SimpleNamespace())  # 音楽の様子が無ければ鳴っていない
    assert not playing(None)
