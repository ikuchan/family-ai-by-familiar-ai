"""環-aw 第 3 段 項目 4：耳の入口の写し（`設計方針_パジュの状態` v0.15 §2.1、§3.1、決定ハ、ガ、ギ、ゾ）。

入口は、開いている、音楽の言葉だけ通す、タイマーの言葉だけ通すの 3 つ。持ち主はタイマーの行と音楽の印のままで、状態図の
入口は写し。写す口（`copy_gate`）は持ち主の理由（`agent.mic_gate_reason`）を受け取り、状態か名前が違うときだけ移るか
写し直す。タイマーの写しは、入口を閉じているタイマーの名前を持つ（決定ゾ）。写すのは、声を受けるとき（門の判定の前）と、
自律機構の輪が回るたび（決定ギ）。状態の行は状態図の入口を読む。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from familiar_agent.agent import MUSIC_GATE_REASON
from familiar_agent.loop.paju_state import EAR_MUSIC, EAR_OPEN, EAR_TIMER, PajuState

PASTA = "タイマー「パスタ」"


def _gate_state(sm: PajuState) -> str:
    ids = {s.id for s in sm.configuration}
    return next((n for n in ("ear_open", "music_only", "timer_only") if n in ids), "")


async def _chart() -> PajuState:
    sm = PajuState()
    await sm.start()
    return sm


def test_the_gate_starts_open():
    async def go():
        sm = await _chart()
        assert _gate_state(sm) == "ear_open" and sm.gate() == EAR_OPEN and sm.gate_timer == ""

    asyncio.run(go())


def test_the_copy_follows_the_owner():
    async def go():
        sm = await _chart()
        await sm.copy_gate(PASTA)
        assert _gate_state(sm) == "timer_only" and sm.gate() == EAR_TIMER
        assert sm.gate_timer == PASTA
        await sm.copy_gate(MUSIC_GATE_REASON)
        assert _gate_state(sm) == "music_only" and sm.gate() == EAR_MUSIC
        assert sm.gate_timer == ""
        await sm.copy_gate("")
        assert _gate_state(sm) == "ear_open" and sm.gate() == EAR_OPEN

    asyncio.run(go())


def test_a_timer_named_after_music_is_still_a_timer():
    """**反証の側**：音楽かは持ち主の理由が `MUSIC_GATE_REASON` と同じかで決める（名前に「音楽」が入っても取り違えない）。"""

    async def go():
        sm = await _chart()
        await sm.copy_gate("タイマー「音楽の練習」")
        assert sm.gate() == EAR_TIMER

    asyncio.run(go())


def test_another_timer_rewrites_the_name_without_moving():
    """状態が同じでも、名前が違えば（別のタイマーに掛け替えた）名前を写し直す（決定ゾ）。移りは送らない。"""

    async def go():
        sm = await _chart()
        await sm.copy_gate(PASTA)
        with patch.object(sm, "_fire", wraps=sm._fire) as fire:
            await sm.copy_gate("タイマー「卵」")
        assert sm.gate_timer == "タイマー「卵」" and _gate_state(sm) == "timer_only"
        fire.assert_not_called()

    asyncio.run(go())


def test_the_same_copy_sends_nothing_and_does_not_wait_for_the_turn():
    """同じなら送らない。輪は 0.5 秒ごとに写すので、順番も待たない。"""

    async def go():
        sm = await _chart()
        await sm.copy_gate(PASTA)
        with patch.object(sm, "_fire", wraps=sm._fire) as fire:
            async with sm._turn():
                await asyncio.wait_for(sm.copy_gate(PASTA), timeout=0.5)
        fire.assert_not_called()

    asyncio.run(go())


# ── 門は、写してから判定する ─────────────────────────────────────────────


def test_the_ear_copies_before_judging():
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    async def go():
        a = _agent(stream_returns=[])
        a.mic_gate_reason = MagicMock(return_value=MUSIC_GATE_REASON)
        ip = InformationProcessing(a)
        assert await ip._ear() == EAR_MUSIC
        assert ip._paju().gate() == EAR_MUSIC
        a.mic_gate_reason = MagicMock(return_value="")
        assert await ip._ear() == EAR_OPEN
        await ip.close()

    asyncio.run(go())


def test_an_unreadable_owner_opens_the_ear():
    """**反証の側**：持ち主が読めなければ、いままでどおり開いている扱い。"""
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    async def go():
        a = _agent(stream_returns=[])
        a.mic_gate_reason = MagicMock(side_effect=RuntimeError("DB"))
        ip = InformationProcessing(a)
        assert await ip._ear() == EAR_OPEN
        await ip.close()

    asyncio.run(go())


# ── 自律機構の輪で写す ───────────────────────────────────────────────────


def test_the_tonic_loop_copies_the_gate():
    from familiar_agent.loop.tonic import Tonic
    from familiar_agent.core.drive_dynamics import DriveFiring
    from familiar_agent.drive_register import AiDrivers

    async def go():
        sm = await _chart()
        agent = SimpleNamespace(
            _music_state=SimpleNamespace(playing=False),
            paju_state=sm,
            mic_gate_reason=lambda: PASTA,
        )
        step = AsyncMock(return_value=(DriveFiring(), AiDrivers()))
        with patch("familiar_agent.loop.tonic.step_drives", new=step):
            t = Tonic(MagicMock(), period=0.01, agent=agent)
            t.start()
            for _ in range(200):
                if sm.gate() == EAR_TIMER:
                    break
                await asyncio.sleep(0.005)
            await t.close()
        assert sm.gate() == EAR_TIMER and sm.gate_timer == PASTA

    asyncio.run(go())


# ── 状態の行は状態図の入口を読む ─────────────────────────────────────────


def _snap(sm: PajuState):
    from familiar_agent.diagnostics import build_gui_diagnostics

    window = MagicMock()
    window._agent_ready = True
    window._agent_running = False
    window._agent_init_failed = False
    window._agent.paju_state = sm
    window._agent.mic_gate_reason = MagicMock(side_effect=AssertionError("持ち主を読まない"))
    window._realtime_stt.gated = False
    return build_gui_diagnostics(window)


def test_the_status_line_names_the_timer_from_the_chart():
    async def go():
        sm = await _chart()
        await sm.copy_gate(PASTA)
        return sm

    snap = _snap(asyncio.run(go()))
    assert snap.headline == "🔇 聞いていない（タイマー「パスタ」 が鳴るまで）"
    assert "mic=closed" in snap.readiness


def test_the_status_line_reads_music_from_the_chart():
    async def go():
        sm = await _chart()
        await sm.copy_gate(MUSIC_GATE_REASON)
        return sm

    snap = _snap(asyncio.run(go()))
    assert snap.headline == "🎵 音楽中：名前か音楽の言葉だけ届く"
    assert "mic=music" in snap.readiness


def test_an_open_gate_is_ready():
    """**反証の側**：入口が開いていれば、聞いていないとは出さない。"""
    snap = _snap(asyncio.run(_chart()))
    assert snap.phase == "ready" and "mic=" not in snap.readiness


def test_the_log_line_names_the_gate():
    """状態図のログの 1 行に入口を出す（2026-11-17 本人の承認）。"""

    async def go():
        sm = await _chart()
        assert "/入口=開いている" in sm.describe()
        await sm.copy_gate(PASTA)
        assert "/入口=タイマーの言葉だけ通す" in sm.describe()

    asyncio.run(go())


def test_a_chart_that_cannot_start_does_not_break_the_ear():
    """**反証の側**：状態図が初めの状態に入れなくても、写す口は例外を出さない（門は今のまま開いている扱い）。"""

    async def go():
        sm = PajuState()
        with patch.object(sm, "start", AsyncMock(side_effect=RuntimeError("start"))):
            await sm.copy_gate(PASTA)
        assert sm.gate() == EAR_OPEN and sm.gate_timer == ""

    asyncio.run(go())
