"""環-aw 第 4 段 項目 1：頭の寝ている（`設計方針_パジュの状態` §3.2、決定ダ、ナ、ク）。

内省のあいだを頭の寝ているにする。自律機構の rest の発火が眠いを置き（`feel_sleepy`）、頭が何もしていないなら眠る
（`fall_asleep`、事象は「眠くなった」`sleep_began`）。考えているか答えを言っているあいだは眠らず、求めを閉じてから眠る
（持ち越し）。内省を終えたら `wake_up` が「起きた」（`woke`）を送り、何もしていないへ戻る。眠いは状態図のデータに持つ。
"""

from __future__ import annotations

import asyncio
import logging

from familiar_agent.loop.paju_state import PajuState


async def _chart() -> PajuState:
    sm = PajuState()
    await sm.start()
    return sm


def test_the_head_sleeps_from_idle_and_wakes():
    async def go():
        sm = await _chart()
        assert not await sm.fall_asleep()  # 眠くなければ眠らない
        sm.feel_sleepy()
        assert await sm.fall_asleep()
        assert sm.head_state() == "sleeping" and not sm.sleepy
        assert "頭=寝ている" in sm.describe()
        await sm.wake_up()
        assert sm.head_state() == "idle"

    asyncio.run(go())


def test_a_sleepy_head_waits_for_the_request_to_close():
    """持ち越し（決定ナ）：考えているあいだは眠らず、眠いは残る。求めを閉じたら眠る。"""

    async def go():
        sm = await _chart()
        await sm.fire("request_opened")
        sm.feel_sleepy()
        assert not await sm.fall_asleep()
        assert sm.head_state() == "thinking" and sm.sleepy
        await sm.fire("request_closed")
        assert await sm.fall_asleep()
        assert sm.head_state() == "sleeping"

    asyncio.run(go())


def test_a_request_does_not_open_while_sleeping(caplog):
    """**反証の側**：寝ているあいだは求めを開く移りが無い。状態図が捨て、1 行残す（決定ユ）。"""

    async def go():
        sm = await _chart()
        sm.feel_sleepy()
        await sm.fall_asleep()
        with caplog.at_level(logging.INFO, logger="familiar_agent.loop.paju_state"):
            await sm.fire("request_opened")
        assert sm.head_state() == "sleeping"

    asyncio.run(go())
    assert "request_opened を受ける移りが無いので捨てた" in caplog.text


def test_an_offer_is_dropped_when_the_window_lapses_while_sleeping():
    """寝ているあいだに窓が閉じたら、何もしていないときと同じく申し出をなしへ戻す（決定ゲ）。"""

    async def go():
        sm = await _chart()
        await sm.fire("window_opened")
        await sm.make_offer("調べようか？", session=1)
        sm.feel_sleepy()
        await sm.fall_asleep()
        await sm.fire("window_lapsed")
        assert not sm.offer_waiting()

    asyncio.run(go())


# ── 自律機構：rest の発火で眠いを置き、輪が回るたびに眠れるかを見る（項目 2）──────────────


def _tonic_run(sm: "PajuState | None", rest, *, until, ticks: int = 400, during=None):
    """rest を 1 回だけ発火させ、`until(rest)` が真になるまで輪を回す。`during` は輪の途中で 1 回だけ呼ぶ。"""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, patch

    from familiar_agent.core.drive_dynamics import DriveFiring
    from familiar_agent.drive_register import AiDrivers
    from familiar_agent.loop.tonic import Tonic

    once = iter([DriveFiring(rest=True)])
    step = AsyncMock(side_effect=lambda *a, **k: (next(once, DriveFiring()), AiDrivers()))
    agent = SimpleNamespace(_music_state=SimpleNamespace(playing=False), paju_state=sm)
    seen: dict = {}

    async def go():
        if sm is not None:
            await sm.start()
        with (
            patch("familiar_agent.loop.tonic.step_drives", new=step),
            patch("familiar_agent.loop.tonic.run_rest_pass", new=rest),
        ):
            t = Tonic(MagicMock(), period=0.01, agent=agent)
            t.start()
            for i in range(ticks):
                if until(rest):
                    break
                if during is not None and i == 20:
                    await during(seen)
                await asyncio.sleep(0.005)
            await t.close()
        seen["steps"] = step.await_count

    asyncio.run(go())
    return seen


def test_the_tonic_sleeps_for_the_rest_pass_and_wakes():
    from unittest.mock import AsyncMock

    sm = PajuState()
    heads: list[str] = []
    rest = AsyncMock(side_effect=lambda agent: heads.append(sm.head_state()) or "内省した")
    _tonic_run(sm, rest, until=lambda r: r.await_count > 0 and sm.head_state() == "idle")
    assert heads == ["sleeping"]  # 内省のあいだは寝ている
    assert sm.head_state() == "idle" and not sm.sleepy  # 終えたら起きる


def test_the_tonic_wakes_even_when_the_rest_pass_fails():
    from unittest.mock import AsyncMock

    sm = PajuState()
    rest = AsyncMock(side_effect=RuntimeError("内省"))
    _tonic_run(sm, rest, until=lambda r: r.await_count > 0 and sm.head_state() == "idle")
    assert rest.await_count == 1 and sm.head_state() == "idle"


def test_the_tonic_waits_for_the_request_to_close_and_keeps_turning():
    """**反証の側**：求めが開いているあいだは内省を始めず、輪は回り続ける（タイマーの見張りを遅らせない・決定ダ）。"""
    from unittest.mock import AsyncMock

    sm = PajuState()
    rest = AsyncMock(return_value="内省した")

    async def opened_first():
        await sm.start()
        await sm.fire("request_opened")  # 発火の前に求めを開いておく

    asyncio.run(opened_first())

    async def close_later(seen):
        await asyncio.sleep(0.2)
        seen["ran_while_open"] = rest.await_count
        seen["sleepy_while_open"] = sm.sleepy
        await sm.fire("request_closed")

    seen = _tonic_run(sm, rest, until=lambda r: r.await_count > 0, during=close_later)
    assert seen.get("ran_while_open") == 0 and seen.get("sleepy_while_open")
    assert seen["steps"] > 10  # 待つあいだも輪は回った
    assert rest.await_count == 1


def test_without_a_chart_the_tonic_runs_the_rest_pass_at_once():
    """状態図の無い構成（試験の装置など）では、いままでどおりすぐ内省する。"""
    from unittest.mock import AsyncMock

    rest = AsyncMock(return_value="内省した")
    _tonic_run(None, rest, until=lambda r: r.await_count > 0)
    assert rest.await_count == 1


# ── 捨てる口：寝ているあいだは、声も打った言葉も命令も、画面で知らせずに捨てる（項目 3・決定ク、ヨ、ダ）──────


def _agent(sm: PajuState):
    from unittest.mock import AsyncMock, MagicMock

    from familiar_agent.agent import EmbodiedAgent

    a = EmbodiedAgent.__new__(EmbodiedAgent)
    a.paju_state = sm
    a._heard_listener = MagicMock()
    a._handle_voice_command = MagicMock(return_value=None)
    a._enroll_voice = AsyncMock(return_value=None)
    a._handle_speaker_command = MagicMock(return_value=None)
    a._handle_timer_command = AsyncMock(return_value="止めた")
    a._info_processing = MagicMock()
    a._info_processing.push_utterance = AsyncMock(return_value="はい")
    return a


async def _asleep() -> PajuState:
    sm = await _chart()
    sm.feel_sleepy()
    await sm.fall_asleep()
    return sm


def test_words_and_commands_are_dropped_while_sleeping(caplog):
    async def go():
        a = _agent(await _asleep())
        with caplog.at_level(logging.INFO, logger="familiar_agent.agent"):
            for text in ("パジュ、いま何時？", "/timer stop", "/mic on"):
                assert await a.run(text) == ""
        a._handle_timer_command.assert_not_awaited()  # 命令も捨てる（2026-11-17 本人の決定）
        a._info_processing.push_utterance.assert_not_awaited()
        a._heard_listener.assert_not_called()  # 画面で知らせない（決定ヨ）

    asyncio.run(go())
    assert caplog.text.count("寝ているので捨てた") == 3
    assert "何時" not in caplog.text  # 言葉の中身はログに出さない


def test_a_command_is_handled_while_awake():
    """**反証の側**：起きていれば、命令はいままでどおり効き、画面に受けたと知らせる。"""

    async def go():
        a = _agent(await _chart())
        assert await a.run("/timer stop") == "止めた"
        a._handle_timer_command.assert_awaited_once()
        a._heard_listener.assert_called_once()

    asyncio.run(go())


# ── 状態の行：寝ているあいだは「寝ている（内省中）」と出す（項目 4・決定ダ）────────────────────


def _snap(sm: PajuState):
    from unittest.mock import MagicMock

    from familiar_agent.diagnostics import build_gui_diagnostics

    window = MagicMock()
    window._agent_ready = True
    window._agent_running = False
    window._agent_init_failed = False
    window._agent.paju_state = sm
    window._realtime_stt.gated = False
    return build_gui_diagnostics(window)


def test_the_status_line_says_sleeping():
    snap = _snap(asyncio.run(_asleep()))
    assert snap.phase == "sleeping"
    assert snap.headline == "😴 寝ている（内省中）"
    assert "命令" in snap.detail


def test_an_awake_head_is_ready():
    """**反証の側**：起きていれば、寝ているとは出さない。"""
    snap = _snap(asyncio.run(_chart()))
    assert snap.phase == "ready" and snap.headline == "Ready"


# ── 眠る前に受けていた求めは、起きるまで待たせてから始める（案 A′・2026-11-18 本人の決定）────────────────
#
# 眠ると求めを開くは、状態図の順番の中で決める。どちらが先でも、寝ていると考えているは同時に成り立たない。
# 捨てないのは、機器の知らせが積む前に済ませていることがあるから（タイマーは鳴った印を付けてから積む）。


def test_opening_a_request_waits_while_sleeping():
    async def go():
        sm = await _asleep()
        opening = asyncio.create_task(sm.open_request("request_opened"))
        await asyncio.sleep(0.1)
        assert not opening.done() and sm.head_state() == "sleeping"
        await sm.wake_up()
        await asyncio.wait_for(opening, timeout=2)
        ids = {s.id for s in sm.configuration}
        assert sm.head_state() == "thinking" and "deaf" in ids

    asyncio.run(go())


def test_opening_a_request_while_awake_is_at_once():
    """**反証の側**：起きていれば、待たずに開く。話しかけない求めは耳を閉じない（決定ザ）。"""

    async def go():
        sm = await _chart()
        await asyncio.wait_for(sm.open_request("quiet_request_opened"), timeout=0.2)
        ids = {s.id for s in sm.configuration}
        assert sm.head_state() == "thinking" and "hearing" in ids

    asyncio.run(go())


def test_sleeping_and_opening_never_hold_together():
    """眠る口と開く口を競らせる。どちらが先でも、片方だけが成り立つ。"""

    async def go(first_sleep: bool):
        sm = await _chart()
        sm.feel_sleepy()
        opening = sm.open_request("request_opened")
        sleeping = sm.fall_asleep()
        pair = (sleeping, opening) if first_sleep else (opening, sleeping)
        tasks = [asyncio.create_task(c) for c in pair]
        await asyncio.sleep(0.1)
        asleep_task = tasks[0] if first_sleep else tasks[1]
        if first_sleep:
            assert asleep_task.done() and asleep_task.result() is True
            assert sm.head_state() == "sleeping"
            await sm.wake_up()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
        assert sm.head_state() == "thinking"
        if not first_sleep:
            assert asleep_task.result() is False and sm.sleepy  # 眠いは持ち越す

    asyncio.run(go(True))
    asyncio.run(go(False))


def _ip():
    from unittest.mock import AsyncMock, MagicMock

    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent as _loop_agent

    a = _loop_agent(stream_returns=[])
    a._oif.write = AsyncMock(return_value="obs-1")
    a._spawn_background_task = MagicMock()
    ip = InformationProcessing(a)
    ip._notify_request_state = MagicMock()
    return ip


def test_a_device_request_begun_while_sleeping_waits_for_the_wake():
    """タイマーの知らせなど、眠る前に積まれた求めは、捨てずに起きてから始める。起点も起きてから書く。"""

    async def go():
        ip = _ip()
        sm = ip._paju()
        await sm.start()
        sm.feel_sleepy()
        await sm.fall_asleep()
        begin = asyncio.create_task(ip._begin_request(kind="機器", text="[タイマー] 卵"))
        await asyncio.sleep(0.1)
        assert not begin.done()
        ip._agent._oif.write.assert_not_awaited()
        from datetime import datetime, timezone

        woke_at = datetime.now(timezone.utc)
        await sm.wake_up()
        await asyncio.wait_for(begin, timeout=2)
        assert sm.head_state() == "thinking"
        ip._agent._oif.write.assert_awaited_once()
        # 言った瞬間（タイマーの起点）は待つ前に取る。起きたあとに取ると「3 分後」が内省の長さだけ遅れる
        assert ip._req.began_at < woke_at

    asyncio.run(go())


def test_a_broken_chart_does_not_stop_a_request():
    """**反証の側**：状態図が落ちても、求めは始める（会話を止めない）。"""
    from unittest.mock import AsyncMock, patch

    async def go():
        ip = _ip()
        sm = ip._paju()
        with patch.object(sm, "open_request", AsyncMock(side_effect=RuntimeError("chart"))):
            await asyncio.wait_for(ip._begin_request(kind="機器", text="[タイマー] 卵"), timeout=1)
        ip._agent._oif.write.assert_awaited_once()

    asyncio.run(go())


def test_the_tonic_does_not_sleep_while_music_plays():
    """音楽が鳴っているあいだは眠らず、眠いを持ち越す。止まって頭が何もしていなければ眠る（環-ba・2026-11-18 本人の決定）。"""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, patch

    from familiar_agent.core.drive_dynamics import DriveFiring
    from familiar_agent.drive_register import AiDrivers
    from familiar_agent.loop.tonic import Tonic

    sm = PajuState()
    rest = AsyncMock(return_value="内省した")
    music = SimpleNamespace(playing=True)
    agent = SimpleNamespace(_music_state=music, paju_state=sm)
    step = AsyncMock(return_value=(DriveFiring(), AiDrivers()))

    async def go():
        await sm.start()
        sm.feel_sleepy()  # 鳴る前の rest の発火で置いた眠い
        with (
            patch("familiar_agent.loop.tonic.step_drives", new=step),
            patch("familiar_agent.loop.tonic.run_rest_pass", new=rest),
        ):
            t = Tonic(MagicMock(), period=0.01, agent=agent)
            t.start()
            for _ in range(30):
                await asyncio.sleep(0.005)
            during = (rest.await_count, sm.sleepy, sm.head_state())
            music.playing = False
            for _ in range(400):
                if rest.await_count:
                    break
                await asyncio.sleep(0.005)
            await t.close()
        return during

    during = asyncio.run(go())
    assert during == (0, True, "idle")  # 鳴っているあいだは眠らず、眠いは残る
    assert rest.await_count == 1  # 止まったら眠って内省した
