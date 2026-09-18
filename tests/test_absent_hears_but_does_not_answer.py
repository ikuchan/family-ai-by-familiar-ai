"""誰も見えていないときの会話入力は、入口で止めて同じ器に溜める（2026-09-17）。

「聞けない理由」は 2 つ——黙っているよう頼まれている（情-h）と、誰も見えない。どちらも
入口で止め（調停も主LLM も回らない）、`silence_hold` の器に溜め、人が映った最初の求めの W に
列挙して主LLM がそのとき判断する。返事を作ってから溜める（環-b の `pending_speech`）のは
機器の知らせだけ。見に行く理由として SEEKING の押し上げも入口で行う。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.core.silence_hold import Heard, render
from familiar_agent.loop.event_loop import InformationProcessing, Trigger


def _ip(*, present: float):
    a = MagicMock()
    a._pmm.presence_status = MagicMock(return_value=[])
    a._oif.write = AsyncMock(return_value="obs-1")
    a._observation_perspective = MagicMock(return_value={})
    a._conversation_perspective = MagicMock(return_value={})
    a._social_presence_permission = MagicMock(return_value=present)
    a._nudge_seeking = AsyncMock()
    ip = InformationProcessing(a)
    ip._load_silence = lambda: None  # 黙ってはいない
    return ip, a


def _run(coro):
    async def bounded():
        return await asyncio.wait_for(coro, timeout=2.0)

    return asyncio.run(bounded())


def test_a_voice_with_nobody_visible_is_heard_but_not_answered():
    ip, a = _ip(present=0.0)

    async def scenario():
        fut = asyncio.get_running_loop().create_future()
        swallowed = await ip._swallow_if_unheard(
            Trigger(kind="会話入力", query="こんにちは", future=fut)
        )
        return swallowed, fut

    swallowed, fut = _run(scenario())
    assert swallowed is True
    assert fut.done() and fut.result() == ""
    assert "誰も見えないあいだに聞いた" in a._oif.write.call_args.args[0].content
    assert [(h.kind, h.why) for h in ip._muted] == [("会話入力", "誰も見えなかった")]
    a._nudge_seeking.assert_awaited_once()  # 見に行く理由にはなる


def test_device_and_affect_still_go_to_the_exit_gate_when_absent():
    ip, a = _ip(present=0.0)
    d = _run(ip._swallow_if_unheard(Trigger(kind="機器", query="タイマー", result="時間")))
    u = _run(ip._swallow_if_unheard(Trigger(kind="情動", query="seeking")))
    assert (d, u) == (False, False)
    a._nudge_seeking.assert_not_awaited()


def test_when_someone_appears_the_heard_things_ride_the_next_request():
    ip, a = _ip(present=0.0)
    _run(ip._swallow_if_unheard(Trigger(kind="会話入力", query="こんにちは")))
    a._social_presence_permission = MagicMock(return_value=1.0)
    swallowed = _run(
        ip._swallow_if_unheard(Trigger(kind="機器", query="入室", result="誰か が来た"))
    )
    assert swallowed is False
    assert [h.text for h in ip._pending_heard] == ["こんにちは"] and ip._muted == []


def test_the_heading_says_why_it_could_not_answer():
    now = time.time()
    items = [Heard(kind="会話入力", text="こんにちは", at=now, why="誰も見えなかった")]
    text = render(items, since=now, until=now, max_chars=500)
    assert text.startswith("誰も見えなかったあいだ（")
    both = items + [Heard(kind="会話入力", text="静かにして", at=now)]
    assert render(both, since=now, until=now, max_chars=500).startswith(
        "黙っていた／誰も見えなかったあいだ（"
    )


# ── 情-k：不在で溜めたものは「沈黙が明けた」ではない（2026-09-18 12:38 実機）─────


def test_things_heard_while_absent_do_not_trigger_silence_lifted():
    ip, a = _ip(present=0.0)
    ip.push_device = MagicMock()
    _run(ip._swallow_if_unheard(Trigger(kind="会話入力", query="こんにちは")))
    ip.check_silence_lifted()  # 黙ってはいない・器には不在の 1 件だけ
    ip.push_device.assert_not_called()


def test_things_heard_while_silent_still_trigger_it_when_it_lifts():
    from familiar_agent.core.silence_hold import Heard

    ip, a = _ip(present=1.0)
    ip.push_device = MagicMock()
    ip._muted = [Heard(kind="会話入力", text="明日の予定は？", at=time.time(), why="黙っていた")]
    ip.check_silence_lifted()  # 依頼は消えている（`_load_silence` は None）
    ip.push_device.assert_called_once()


# ── タイマーがあるときの操作の言葉は、誰も見えなくても通す（出-ab・2026-09-18）────────
#
# 鳴っている最中に映らない位置で「止めて」→ `誰も見えないので聞くだけ`（実機 22:13:04）。沈黙の門は
# 情-m で操作の言葉を通すようにしたが、鳴った瞬間に沈黙は解け、代わりに不在の門が効いていた。
# `[タイマー]` が動いている／一時停止中／鳴っているときは、操作の言葉（`is_control_word`）を通す。


def _ip_with_timer(*, frame: str, ringing: bool = False):
    ip, a = _ip(present=0.0)
    a._timer_tool.frame = MagicMock(return_value=frame)
    a._dif.ringing = ringing
    return ip, a


def test_a_control_word_passes_the_absent_gate_while_a_timer_runs():
    ip, _ = _ip_with_timer(frame="[タイマー]\n- id=1 パスタ 鳴っている")
    assert _run(ip._swallow_if_unheard(Trigger(kind="会話入力", query="止めて"))) is False
    assert ip._muted == []


def test_ordinary_talk_is_still_swallowed_while_a_timer_runs():
    ip, _ = _ip_with_timer(frame="[タイマー]\n- id=1 パスタ 残り 2:00")
    assert _run(ip._swallow_if_unheard(Trigger(kind="会話入力", query="こんにちは"))) is True


def test_a_control_word_is_swallowed_when_no_timer_exists():
    ip, _ = _ip_with_timer(frame="")
    assert _run(ip._swallow_if_unheard(Trigger(kind="会話入力", query="止めて"))) is True


def test_the_ring_alone_counts_as_a_timer():
    ip, _ = _ip_with_timer(frame="", ringing=True)
    assert _run(ip._swallow_if_unheard(Trigger(kind="会話入力", query="止めて"))) is False
