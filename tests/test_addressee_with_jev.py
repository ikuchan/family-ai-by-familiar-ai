"""窓の中の名前の無い声が、パジュ宛てか家族どうしかを Jev が決める（出-au 段 5-2・2026-09-27・`設計方針_判定の段` §2.2.3）。

窓（30 秒）の中なら名前が無くても受けていたので、家族どうしの話（「ごはんできたよー」）にも返事をしていた。

- 窓の中で名前の無い**声**のときだけ、その言葉・直近のやりとり・顔ぶれを送って聞く。名前がある入力とキーボードは聞かない
  （パジュに向けたものと決まっている）。
- **家族どうし**なら捨てる。窓の外の入力と同じく、打ち切り・時刻の印・会話ログ・窓の延長のどれも起こさない。
- パジュ宛て・分からない・Jev が使えない・確信度 0.6 未満なら受ける（倒し先）。
- 窓の延長は判定の後（家族どうしの話では延ばさない）。
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.core import jev_judges
from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent

pytestmark = pytest.mark.real_window  # 門そのものを確かめる


def _answer(pick: str, conf: float = 0.8) -> JevAnswer:
    return JevAnswer(ok=True, answers={"to_whom": {"choice": pick, "confidence": conf}})


def _ip(answer=None, *, exc=None):
    a = _agent(stream_returns=[])
    a._nudge_seeking = AsyncMock()
    a._occupancy = MagicMock(return_value=1.0)
    a._last_human_at = None
    client = MagicMock()
    client.available = True
    client.ask = AsyncMock(side_effect=exc, return_value=answer)
    a._jev = client
    a.config.jev_confidence_min = 0.6
    ip = InformationProcessing(a)
    ip._load_silence = lambda: None
    ip._ensure_driver = lambda: None
    ip._abort_lookups = AsyncMock()
    heard: list = []
    ip.set_heard_listener(lambda text, ok: heard.append(ok))
    return ip, a, heard


def _push(ip, text, *, source="voice", arrived=None):
    async def go():
        task = asyncio.ensure_future(ip.push_utterance(text, source=source, arrived=arrived))
        await asyncio.sleep(0.01)
        queued = not task.done()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return queued

    return asyncio.run(go())


def test_family_talk_inside_the_window_is_dropped_without_side_effects():
    ip, a, heard = _ip(_answer("family"))
    ip._wake_window().open(100.0)  # 窓は 130 まで
    assert _push(ip, "ごはんできたよー", arrived=110.0) is False
    assert heard == [False]
    assert a._last_human_at is None
    assert not ip._wake_window().is_open(135.0)  # 延ばしていない（延ばせば 140 まで開く）


@pytest.mark.parametrize(
    "answer",
    [
        _answer("paju"),
        _answer("unknown"),
        _answer("family", 0.5),
        JevAnswer(ok=False, error="時間切れ"),
    ],
)
def test_otherwise_it_is_heard_and_extends_the_window(answer):
    ip, a, heard = _ip(answer)
    ip._wake_window().open(100.0)
    assert _push(ip, "明日の天気は？", arrived=105.0) is True
    assert heard == [True]
    assert ip._wake_window().is_open(112.0)  # 受けたので 115 まで延びた（窓 10 秒）


def test_named_input_and_typing_are_not_asked():
    ip, a, _ = _ip(_answer("family"))
    assert _push(ip, "パジュ、明日の天気は？") is True
    ip._wake_window().open(time.monotonic())
    assert _push(ip, "明日の天気は？", source="keyboard") is True
    a._jev.ask.assert_not_awaited()


def test_the_state_carries_the_words_and_who_is_here():
    c = MagicMock()
    c.available = True
    c.ask = AsyncMock(return_value=_answer("paju"))
    got = asyncio.run(
        jev_judges.judge_addressee(
            c,
            text="ごはんできたよー",
            recent="- 相手：パジュ、今日の予定は？",
            present="パパ・たいき",
            min_conf=0.6,
        )
    )
    assert got == jev_judges.TO_PAJU
    state, questions = c.ask.await_args.args
    assert "ごはんできたよー" in state and "今日の予定" in state and "たいき" in state
    assert set(questions["to_whom"]["criteria"]) == {"paju", "family", "unknown"}
