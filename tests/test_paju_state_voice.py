"""環-aw 第 5 段：口の声（`設計方針_パジュの状態` §3.5）。

口に入れ子の領域「声」（`voice`）を置く。状態は、なし（`no_voice`）、考え中の音（`thinking_cue`）、合図
（`cue`：効果音B・はいっ・えっと・黙る音）、話している（`speaking`：パジュの声のすべて）の 4 つ。送り元は dif だけ。
"""

from __future__ import annotations

import asyncio
import logging
import time

from familiar_agent.loop.paju_state import PajuState


async def _chart() -> PajuState:
    sm = PajuState()
    await sm.start()
    return sm


def test_the_voice_starts_with_no_voice():
    async def go():
        sm = await _chart()
        assert sm.voice_state() == "no_voice"
        assert "声=なし" in sm.describe()

    asyncio.run(go())


def test_the_thinking_cue_comes_and_goes():
    async def go():
        sm = await _chart()
        await sm.fire("thinking_cue_began")
        assert sm.voice_state() == "thinking_cue"
        assert "声=考え中の音" in sm.describe()
        await sm.fire("thinking_cue_ended")
        assert sm.voice_state() == "no_voice"

    asyncio.run(go())


def test_a_cue_comes_from_no_voice_or_the_thinking_cue_and_re_enters_itself():
    async def go():
        sm = await _chart()
        await sm.fire("cue_began")
        assert sm.voice_state() == "cue"
        await sm.fire("cue_began")  # 合図 → 合図（外へ出て入り直す）
        assert sm.voice_state() == "cue"
        assert "声=合図" in sm.describe()
        await sm.fire("cue_ended")
        assert sm.voice_state() == "no_voice"
        await sm.fire("thinking_cue_began")
        await sm.fire("cue_began")
        assert sm.voice_state() == "cue"

    asyncio.run(go())


def test_speaking_comes_from_any_other_voice():
    async def go():
        sm = await _chart()
        for before in ("", "thinking_cue_began", "cue_began"):
            if before:
                await sm.fire(before)
            await sm.fire("voice_began")
            assert sm.voice_state() == "speaking"
            assert "声=話している" in sm.describe()
            await sm.fire("voice_ended", played=True)
            assert sm.voice_state() == "no_voice"

    asyncio.run(go())


def test_the_voice_region_leaves_the_other_regions_alone():
    """**反証の側**：声の事象で、頭、耳、黙るかは動かない（並行の領域の中で状態が二重にならない）。"""

    async def go():
        sm = await _chart()
        before = sm.describe()
        await sm.fire("cue_began")
        await sm.fire("cue_began")
        await sm.fire("voice_began")
        assert sm.head_state() == "idle" and not sm.is_deaf() and not sm.is_silent()
        ids = [s.id for s in sm.configuration]
        assert len(ids) == len(set(ids))
        assert sum(1 for n in ("no_voice", "thinking_cue", "cue", "speaking") if n in ids) == 1
        await sm.fire("voice_ended", played=False)
        assert sm.describe() == before

    asyncio.run(go())


def test_an_old_thinking_cue_end_does_not_end_a_cue(caplog):
    """**反証の側**：合図のときに届いた古い「考え中の音が終わった」は移りが無く、捨てられて 1 行残る。合図のまま。"""

    async def go():
        sm = await _chart()
        await sm.fire("thinking_cue_began")
        await sm.fire("cue_began")
        with caplog.at_level(logging.INFO, logger="familiar_agent.loop.paju_state"):
            await sm.fire("thinking_cue_ended")
        assert sm.voice_state() == "cue"

    asyncio.run(go())
    assert any("thinking_cue_ended を受ける移りが無い" in r.getMessage() for r in caplog.records)


def test_speech_does_not_end_without_speaking(caplog):
    """**反証の側**：話していないときの「話し終えた」と、話しているあいだの合図は捨てる。"""

    async def go():
        sm = await _chart()
        with caplog.at_level(logging.INFO, logger="familiar_agent.loop.paju_state"):
            await sm.fire("voice_ended", played=True)
            await sm.fire("voice_began")
            await sm.fire("cue_began")
            await sm.fire("thinking_cue_began")
        assert sm.voice_state() == "speaking"

    asyncio.run(go())
    dropped = [r.getMessage() for r in caplog.records if "移りが無い" in r.getMessage()]
    assert len(dropped) == 3


# ── 項目 3：窓を声の状態から保つ ──────────────────────────────────────────


def _window(go, monkeypatch, after_sec: float = 0.2):
    """話し終えてから閉じるまでを `after_sec` 秒に縮めて回す（本物は 10 秒）。"""
    from familiar_agent.loop import paju_state

    monkeypatch.setattr(paju_state, "WAKE_WINDOW_SEC", after_sec)

    async def run():
        sm = PajuState()
        await sm.start()
        await go(sm)

    asyncio.run(asyncio.wait_for(run(), timeout=5.0))


def _almost_lapsed(left: float) -> float:
    """いまから `left` 秒で切れる窓の、開けた時刻（窓のデータは本物の 10 秒で数える）。"""
    from familiar_agent.core.wake_window import WAKE_WINDOW_SEC

    return time.monotonic() - WAKE_WINDOW_SEC + left


def _ids(sm: PajuState) -> set[str]:
    return {s.id for s in sm.configuration}


def test_a_long_voice_keeps_the_open_window_open_past_its_end(monkeypatch):
    """窓が開いているときに話し始めたら、`until` を過ぎても窓は開いていて、門も開いていると答える。話し終えてから閉じる。"""

    async def go(sm):
        await sm.open_window(_almost_lapsed(0.1))
        await sm.fire("voice_began")
        await asyncio.sleep(0.4)  # 元の窓はもう切れている
        assert "open" in _ids(sm)
        assert sm.window_open_at(time.monotonic())
        await sm.fire("voice_ended", played=True)
        assert "open" in _ids(sm) and sm.window_open_at(time.monotonic())
        await asyncio.sleep(0.4)  # 話し終えてから 0.2 秒（本物は 10 秒）
        assert "shut" in _ids(sm)
        assert not sm.window_open_at(time.monotonic())

    _window(go, monkeypatch)


def test_a_name_heard_while_speaking_continues_the_same_talk(monkeypatch):
    """話している途中で名前を聞いて開け直しても、窓の会話の番号は進まない。"""

    async def go(sm):
        await sm.open_window(_almost_lapsed(0.1))
        session = sm.wake.session
        await sm.fire("voice_began")
        await asyncio.sleep(0.3)
        await sm.open_window(time.monotonic())
        await sm.extend_window(time.monotonic())
        assert sm.wake.session == session
        await asyncio.sleep(0.3)
        assert "open" in _ids(sm)  # 開け直しても、話しているあいだは切れない
        await sm.fire("voice_ended", played=True)

    _window(go, monkeypatch)


def test_a_window_opened_while_speaking_stays_open_until_the_voice_ends(monkeypatch):
    """窓が閉じていて、話している途中で名前を聞いたら、そこから話し終えるまで開いていて、話し終えてから閉じる。"""

    async def go(sm):
        await sm.fire("voice_began")
        await sm.open_window(_almost_lapsed(0.1))
        await asyncio.sleep(0.4)
        assert "open" in _ids(sm)
        await sm.fire("voice_ended", played=True)
        await asyncio.sleep(0.4)
        assert "shut" in _ids(sm)

    _window(go, monkeypatch)


def test_a_voice_does_not_open_a_shut_window(monkeypatch):
    """**反証の側**：窓が閉じているときに話し始めても、窓は開かない。話し終えても開かない。"""

    async def go(sm):
        await sm.fire("voice_began")
        assert "shut" in _ids(sm) and not sm.window_open_at(time.monotonic())
        await sm.fire("voice_ended", played=True)
        assert "shut" in _ids(sm) and not sm.window_open_at(time.monotonic())

    _window(go, monkeypatch)


def test_a_window_that_was_closed_stays_closed_after_the_voice(monkeypatch):
    """**反証の側**：話している途中で窓を閉じたら（`WakeWindow.close`）、話し終えても開け直さない。"""

    async def go(sm):
        await sm.open_window(time.monotonic())
        await sm.fire("voice_began")
        sm.wake.close()
        await sm.fire("voice_ended", played=True)
        assert not sm.window_open_at(time.monotonic())

    _window(go, monkeypatch)


# ── 項目 4：自分の声の見張りは声の状態から読む ────────────────────────────


def test_the_chart_tells_the_end_of_a_voice_with_played():
    """話しているを出るとき、鳴ったか（`played`）を話し終えた口へ渡す（見張りの 1.2 秒・事象ではなく呼び出し）。"""

    async def go():
        ended: list[bool] = []
        sm = PajuState(on_speech_ended=ended.append)
        await sm.start()
        await sm.fire("voice_began")
        await sm.fire("voice_ended", played=True)
        await sm.fire("voice_began")
        await sm.fire("voice_ended", played=False)
        await sm.fire("cue_began")
        await sm.fire("cue_ended")  # 合図は声ではない
        assert ended == [True, False]

    asyncio.run(go())


def test_the_loop_attaches_the_shared_guard_to_its_chart():
    """ループが状態図を作ると、共有の見張りが話しているをその状態図から読む。"""
    from familiar_agent.loop.event_loop import InformationProcessing
    from familiar_agent.voice_guard import get_shared_voice_guard

    from tests.test_event_loop import _agent

    async def go():
        ip = InformationProcessing(_agent(stream_returns=[]))
        sm = ip._paju()
        await sm.start()
        guard = get_shared_voice_guard()
        assert guard.speaking is False
        await sm.fire("voice_began")
        assert guard.speaking is True
        await sm.fire("voice_ended", played=False)
        assert guard.speaking is False

    asyncio.run(go())


# ── 項目 6：黙っているあいだは、求めの外の声を鳴らさない ──────────────────


def _loop_with_speak(result):
    """`dif.speak` を偽にしたループ。渡された印（`through_silence`）を残す。"""
    from unittest.mock import AsyncMock

    from familiar_agent.loop.event_loop import InformationProcessing

    from tests.test_event_loop import _agent

    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._dif.speak = AsyncMock(return_value=result)  # type: ignore[method-assign]
    ip._dif.say_recorded = AsyncMock(return_value=result)  # type: ignore[method-assign]
    return ip


def test_a_voice_in_an_open_request_passes_through_silence():
    """求めが開いているあいだの声（タイマーを掛けた返事、「あと何分？」への返事、鳴った知らせ）は印を付けて渡す。"""

    async def go():
        for kind, text in (("発話", "あと何分？"), ("機器", "[タイマー] 3分")):
            ip = _loop_with_speak(True)
            sm = ip._paju()
            await sm.start()
            ip._req.trigger_kind = kind
            ip._req.request_text = text
            await ip._to_state("request_opened")
            assert await ip._say_aloud("返事") is True
            assert ip._dif.speak.await_args.kwargs["through_silence"] is True
            assert await ip._say_aloud("", recording="timeout") is True
            assert ip._dif.say_recorded.await_args.kwargs["through_silence"] is True
            await ip.close()

    asyncio.run(go())


def test_a_voice_after_the_request_closed_does_not_pass_through_silence():
    """**反証の側**：頭が何もしていない（求めを閉じたあと）の声には印を付けない。"""

    async def go():
        ip = _loop_with_speak(True)
        await ip._paju().start()
        await ip._say_aloud("もっと詳しく調べましょうか？")
        assert ip._dif.speak.await_args.kwargs["through_silence"] is False
        await ip.close()

    asyncio.run(go())


def test_a_held_offer_is_not_kept_and_opens_no_window():
    """黙っていて鳴らさなかった申し出は預けず、窓も開けない（dif が None を返す）。"""

    async def go():
        ip = _loop_with_speak(None)
        sm = ip._paju()
        await sm.start()
        ip._req.trigger_kind = "発話"
        ip._req.named = True  # 名前で受けた話しかけの返事なら、ふだんは話し終わりに窓を開ける
        assert await ip._say_aloud("申し出") is False
        assert not sm.window_open_at(time.monotonic())
        await ip._offer_deep({"request": "x", "found": []})
        assert not sm.offer_waiting()
        assert not sm.window_open_at(time.monotonic())
        await ip.close()

    asyncio.run(go())


def test_a_voice_that_failed_still_counts_as_said():
    """**反証の側**：鳴らそうとして鳴らなかった声（偽）は、いままでどおり言ったことにする（申し出を預ける）。"""

    async def go():
        ip = _loop_with_speak(False)
        sm = ip._paju()
        await sm.start()
        ip._req.trigger_kind = "発話"
        await ip._offer_deep({"request": "x", "found": []})
        assert sm.offer_waiting()
        await ip.close()

    asyncio.run(go())


def test_the_thinking_cue_can_start_while_a_cue_sounds():
    """合図の最中に考え中の音が始まれば考え中の音へ移り、古い合図の終わりでは「なし」へ戻らない（見直しの指摘）。"""

    async def go():
        sm = await _chart()
        await sm.fire("cue_began")
        await sm.fire("thinking_cue_began")
        assert sm.voice_state() == "thinking_cue"
        await sm.fire("cue_ended")  # 前の合図の終わりが遅れて届く
        assert sm.voice_state() == "thinking_cue"
        await sm.fire("thinking_cue_ended")
        assert sm.voice_state() == "no_voice"

    asyncio.run(go())


# ── 黙っているあいだ話さない（環-bb・Lean で証明する決まり・2026-11-21 本人の決定） ──────────────────


def test_an_outside_voice_does_not_start_speaking_while_silenced():
    """黙る依頼が効いているあいだ、求めの外の声（through_silence なし）では、声は話しているへ移らない。"""
    import asyncio
    import time

    from familiar_agent.loop.paju_state import PajuState
    from familiar_agent.silence_state import SilenceRequest

    async def go():
        sm = PajuState()
        await sm.start()
        await sm.hush_with(SilenceRequest(person="", until=time.time() + 600, reason=""))
        await sm.fire("voice_began")
        outside = sm.voice_state()
        await sm.fire("voice_began", through_silence=True)
        inside = sm.voice_state()
        return outside, inside

    outside, inside = asyncio.run(go())
    assert outside == "no_voice"  # 求めの外の声は移らない
    assert inside == "speaking"  # 開いている求めの声は通す
