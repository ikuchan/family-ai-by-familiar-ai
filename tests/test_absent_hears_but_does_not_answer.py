"""入口の門のうち、誰が見えるかに関わるもの（2026-09-17 → 出-as 段 3・2026-09-26）。

「誰も見えないと会話入力を止めて溜める」は、出-as でウェイクワードの窓に置き換えた（窓の外の声は
記録せずに捨てる・`test_the_wake_window_gates_voice.py`）。ここに残るのは、機器と情動が入口を通ること、
黙っていたあいだの見出し、明けたときに渡すこと、タイマーの操作の言葉が名前なしで通ること。
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


def test_device_and_affect_still_go_to_the_exit_gate_when_absent():
    ip, a = _ip(present=0.0)
    d = _run(ip._swallow_if_unheard(Trigger(kind="機器", query="タイマー", result="時間")))
    u = _run(ip._swallow_if_unheard(Trigger(kind="情動", query="seeking")))
    assert (d, u) == (False, False)
    a._nudge_seeking.assert_not_awaited()


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


def test_things_heard_while_silent_do_not_start_anything_when_it_lifts():
    """明けても何もしない（出-as §2.7）。聞いたことは次の会話の求めに乗る。"""
    from familiar_agent.core.silence_hold import Heard

    ip, a = _ip(present=1.0)
    ip.push_device = MagicMock()
    ip._muted = [Heard(kind="会話入力", text="明日の予定は？", at=time.time(), why="黙っていた")]
    ip.check_silence_lifted()  # 依頼は消えている（`_load_silence` は None）
    ip.push_device.assert_not_called()
    assert [h.text for h in ip._muted] == ["明日の予定は？"]  # 捨てずに次の求めまで持つ


# ── タイマーがあるときの操作の言葉は、誰も見えなくても通す（出-ab・2026-09-18）────────
#
# 鳴っている最中に映らない位置で「止めて」→ `誰も見えないので聞くだけ`（実機 22:13:04）。出-as 以降は、
# 名前の無い声（窓の外）でも操作の言葉なら通す、の意味になる。沈黙の門は
# 情-m で操作の言葉を通すようにしたが、鳴った瞬間に沈黙は解け、代わりに不在の門が効いていた。
# `[タイマー]` が動いている／一時停止中／鳴っているときは、操作の言葉（`is_control_word`）を通す。


def _ip_with_timer(*, frame: str, ringing: bool = False):
    ip, a = _ip(present=0.0)
    a._timer_tool.frame = MagicMock(return_value=frame)
    a._dif.ringing = ringing
    return ip, a


def test_a_control_word_passes_the_absent_gate_while_a_timer_runs():
    ip, _ = _ip_with_timer(frame="[タイマー]\n- id=1 パスタ 鳴っている")
    assert (
        _run(ip._swallow_if_unheard(Trigger(kind="会話入力", query="止めて", source="voice")))
        is False
    )
    assert ip._muted == []


def test_the_ring_alone_counts_as_a_timer():
    ip, _ = _ip_with_timer(frame="", ringing=True)
    assert (
        _run(ip._swallow_if_unheard(Trigger(kind="会話入力", query="止めて", source="voice")))
        is False
    )
