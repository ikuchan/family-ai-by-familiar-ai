"""待ちのあいだの電子音（出-bu・2026-10-12・本人の決定：間隔 0.5 秒→1 秒）。音楽中も鳴らす（出-cl・2026-10-15・本人の決定カ）。

10/12 22:15「明日の天気は?」では、合図「はい」から最初のつなぎまで 13 秒無音だった（調べもの約 5 秒・調停の聞き直し
約 2 秒・主LLM 約 5 秒）。機械音 A は最初の判断のあいだ（0.5 秒ほど）しか鳴らない。合図が鳴り終わってから答えの声が
始まるまで、機械音 A を間隔をあけて 1 回ずつ鳴らす。
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def _ip(*, interval=0.05):
    a = _agent(stream_returns=[])
    a.config.wait_beep_interval_seconds = interval
    a._music_state = MagicMock(playing=False)
    ip = InformationProcessing(a)
    ip._req.trigger_kind = "発話"
    beeps: list[float] = []

    async def beep(kind):
        beeps.append(asyncio.get_running_loop().time())
        return True

    ip._dif.beep = beep  # type: ignore[method-assign]
    return ip, beeps


def _run(ip, *, seconds, during=None):
    async def go():
        if during is not None:
            await during()
        await ip._to_state("request_opened")  # 考えているの子に入ると鳴り始める（環-aw 決定テ）
        await asyncio.sleep(seconds)
        ip._stop_beeps()
        await ip.close()

    asyncio.run(go())


def test_it_beeps_while_waiting():
    ip, beeps = _ip()
    _run(ip, seconds=0.3)
    assert len(beeps) >= 3


def test_beeps_while_music_plays():
    """つなぎの声をやめたので、音楽中に鳴らさないと待ちのあいだ何も鳴らない（出-cl・本人の決定カ）。音楽は絞らない。"""
    ip, beeps = _ip()
    ip._agent._music_state.playing = True
    _run(ip, seconds=0.3)
    assert len(beeps) >= 3


def test_beeps_rest_while_a_cue_sounds():
    """**反証の側**：合図の声が鳴っているあいだは休む（つなぎの声は出-cl 段 3 で撤去）。"""
    ip, beeps = _ip()

    async def voice():
        ip._cue_sound = asyncio.ensure_future(asyncio.sleep(10))

    _run(ip, seconds=0.3, during=voice)
    assert beeps == []


def test_the_answer_stops_the_beeps(caplog):
    ip, beeps = _ip()
    ip._say_aloud = AsyncMock()  # type: ignore[method-assign]
    ip._delivery_block_reason = lambda: ""  # type: ignore[method-assign]
    after: list[int] = []

    async def go():
        await ip._to_state("request_opened")
        await asyncio.sleep(0.2)
        await ip._speak("明日は晴れです")
        after.append(len(beeps))
        await asyncio.sleep(0.2)
        after.append(len(beeps))
        await ip.close()

    with caplog.at_level("INFO"):
        asyncio.run(go())
    assert after[0] == after[1] and after[0] >= 1
    assert "待ちの電子音" in caplog.text


def test_closing_the_request_stops_the_beeps():
    ip, beeps = _ip()
    counts: list[int] = []

    async def go():
        await ip._to_state("request_opened")
        await asyncio.sleep(0.15)
        ip._stop_wait_watch()  # 求めが閉じる・打ち切られる道はここを通る
        counts.append(len(beeps))
        await asyncio.sleep(0.15)
        counts.append(len(beeps))
        await ip.close()

    asyncio.run(go())
    assert counts[0] == counts[1]


def _kinds(ip, events, *, seconds=0.12):
    """事象を順に送り、そのあいだに鳴らした電子音の種類を返す。"""
    kinds: list[str] = []

    async def beep(kind):
        kinds.append(kind)
        return True

    ip._dif.beep = beep  # type: ignore[method-assign]

    async def go():
        for e in events:
            await ip._to_state(e)
        await asyncio.sleep(seconds)
        ip._stop_beeps()
        await ip.close()

    asyncio.run(go())
    return kinds


def test_opening_a_keyboard_request_starts_the_beeps():
    """キーボード・情動・機器の求めでも鳴らす（環-aw 決定テ）。前は声の合図のあとだけだった。"""
    ip, _ = _ip()
    ip._req.source = "keyboard"
    assert _kinds(ip, ["request_opened"])[:1] == ["recalling"]


def test_no_request_no_beeps():
    """**反証の側**：求めが無ければ、考えているの子に入らないので鳴らない。"""
    ip, _ = _ip()
    assert _kinds(ip, []) == []


def test_each_child_beeps_its_own_kind():
    """考えているの子に入るときの動作で、その子の音を鳴らす（決定テ）。"""
    for event, kind in (
        ("decide_began", "deciding"),
        ("wait_began", "awaiting"),
        ("check_began", "checking"),
    ):
        ip, _ = _ip()
        kinds = _kinds(ip, ["request_opened", event])
        assert kinds and kinds[-1] == kind and set(kinds) <= {"recalling", kind}


def test_noting_after_the_answer_beeps_again():
    """答えを言い終えて書き留めているへ戻ったら、また鳴らす（決定ヤ）。言っているあいだは鳴らさない。"""
    ip, _ = _ip()
    kinds = _kinds(ip, ["request_opened", "speech_began", "speech_ended"])
    assert kinds[-1:] == ["noting"]


def test_no_beeps_while_answering():
    """**反証の側**：答えを言っているあいだは、電子音の仕事が動いていても鳴らさない（決定テ）。"""
    ip, _ = _ip()
    ip._stop_beeps = lambda: None  # type: ignore[method-assign]  # 止め忘れても鳴らないこと
    kinds = _kinds(ip, ["request_opened", "speech_began"], seconds=0.2)
    assert kinds and set(kinds) <= {"recalling"}


def test_beeps_rest_while_the_judging_sound_plays():
    """Jev が判断しているあいだは機械音 A が鳴り続けるので、電子音は重ねずに休む（出-bg）。"""
    ip, _ = _ip()
    ip._req.source = "voice"
    ip._req.trigger_kind = "発話"

    async def forever():
        await asyncio.sleep(10)

    kinds: list[str] = []

    async def beep(kind):
        kinds.append(kind)
        return True

    ip._dif.beep = beep  # type: ignore[method-assign]

    async def go():
        from familiar_agent.io.dif import _Thinking

        ip._dif.start_thinking = lambda: _Thinking(asyncio.ensure_future(forever()))  # type: ignore[method-assign]
        cue = ip._start_reaction_cue([])
        await ip._to_state("request_opened")
        await asyncio.sleep(0.15)
        cue.stop()
        ip._stop_beeps()
        await ip.close()

    asyncio.run(go())
    assert kinds == []


def test_a_cue_no_longer_starts_the_beeps():
    """鳴らし始めは考えているの子に入るときの動作へ移した（決定テ）。合図の決まりは鳴らし始めない。"""
    from familiar_agent.loop.event_loop import _ReactionCue

    ip, _ = _ip()
    started: list[bool] = []
    ip._start_beeps = lambda: started.append(True)  # type: ignore[method-assign]

    async def go():
        ip._dif.play_cue = lambda cue: asyncio.get_running_loop().create_future()  # type: ignore[method-assign]
        _ReactionCue(ip, MagicMock()).decided("search_deferred")
        await ip.close()

    asyncio.run(go())
    assert started == []


def test_the_dif_plays_each_kind_its_own_sound():
    """音は `reaction_cue` の表で決める（環-aw 第 1 段 項目 6）。決めているは機械音 A と同じ音、ほかの 4 つは子ごとの音。"""
    from familiar_agent.core import reaction_cue as rc
    from familiar_agent.io import dif as dif_mod

    played: list = []

    async def play(path, gain, *, quiet=False):
        played.append(path)
        return True

    from familiar_agent.loop.paju_state import CHILDREN

    kinds = CHILDREN
    assert set(rc.BEEP_WAVS) == set(CHILDREN)  # 子を足したら、音の表にも足す
    d = dif_mod.DIF()
    with patch.object(dif_mod, "_play_wav", play):
        for kind in kinds:
            assert asyncio.run(d.beep(kind)) is True
    assert played == [rc.BEEP_WAVS[k] for k in kinds]
    assert rc.BEEP_WAVS["deciding"] == rc.THINKING_WAV
    names = {k: rc.BEEP_WAVS[k].name for k in kinds if k != "deciding"}
    assert names == {
        k: f"react_beep_{k}.wav" for k in names
    }  # 反証の側：4 つの子は機械音 A ではない
    assert all(p.exists() for p in played)


def test_a_quiet_play_logs_at_debug(tmp_path, monkeypatch, caplog):
    import numpy as np
    import soundfile as sf

    from familiar_agent.tools import tts

    monkeypatch.setattr(tts, "_pw_play_binary", lambda: "/usr/bin/pw-play")
    monkeypatch.setattr(tts, "_pw_target", lambda: None)

    class _Proc:
        returncode = 0

        def __init__(self, *a, **k):
            pass

        def communicate(self, timeout=None):
            import time

            time.sleep(0.12)
            return b"", b""

        def kill(self):
            pass

    monkeypatch.setattr(tts.subprocess, "Popen", _Proc)
    p = tmp_path / "b.wav"
    sf.write(str(p), np.zeros(int(0.1 * 24000), dtype="float32"), 24000)

    with caplog.at_level("DEBUG"):
        asyncio.run(tts._play_via_sounddevice(str(p), 1.0, quiet=True))
    lines = [r for r in caplog.records if "声を鳴らした" in r.getMessage()]
    assert lines and all(r.levelname == "DEBUG" for r in lines)


def test_the_interval_comes_from_config():
    from familiar_agent.config import AgentConfig

    with patch.dict(os.environ, {}, clear=True):
        interval = AgentConfig().wait_beep_interval_seconds  # 設定を丸ごと表示させない（鍵が出る）
    assert interval == 1.0  # 0.5 秒は短すぎた（本人 2026-10-13）


# ── 声を出さない求めでは鳴らさない（環-aw 第 1 段・2026-11-09・本人の決定） ─────────────
# 情動と機器から来た求めは、人が居なければ声に出さない（`_delivery_block_reason`）。そのとき電子音だけが空の家に
# 鳴らないようにする。声とキーボードの求めは、門が理由を返さないので、いつも鳴らす。


@pytest.mark.parametrize("kind", ["情動", "機器"])
def test_no_beeps_when_nobody_would_hear_the_answer(kind):
    ip, beeps = _ip()
    ip._req.trigger_kind = kind
    ip._delivery_block_reason = lambda: "聞く相手が居ない"  # type: ignore[method-assign]
    _run(ip, seconds=0.3)
    assert beeps == []


@pytest.mark.parametrize("kind", ["情動", "機器"])
def test_beeps_when_someone_would_hear_the_answer(kind):
    """**反証の側**：人が居れば、情動と機器の求めでも鳴らす。"""
    ip, beeps = _ip()
    ip._req.trigger_kind = kind
    ip._delivery_block_reason = lambda: ""  # type: ignore[method-assign]
    _run(ip, seconds=0.3)
    assert len(beeps) >= 3


def test_a_voice_request_beeps_even_when_the_camera_sees_nobody():
    """**反証の側**：声の求めは在席で止めない（門が理由を返さない・出-as §2.3）。"""
    ip, beeps = _ip()
    ip._agent._someone_here = lambda: False
    _run(ip, seconds=0.3)
    assert len(beeps) >= 3
