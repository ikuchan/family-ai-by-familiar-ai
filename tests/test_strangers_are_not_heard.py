"""知らない声は、門のいちばん先で無条件に捨てる（知-ar・2026-10-11・本人の決定）。

登録した家族以外の声も会話に流れていた（声で照らすのは門を通った後で、誰にも当たらない声は話者を既定の人に戻すだけ）。
知らない声＝声紋があり、声を登録した誰にも基準（`voice_stranger_threshold`）まで届かない声。名前・黙っていての依頼・
窓より前で捨て、窓も照合も触らない。知らない人の気配として SAFETY は押し上げる（誰か居るときだけ・本人の決定ア）。
声紋が無い声・キーボード・登録が 1 つも無いときは判断できないので通す。基準は登録と正解付きの点数が揃ってから決める
（未設定なら効かない）。
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from familiar_agent.loop.event_loop import Trigger
from tests.test_voice_picks_the_speaker import PAPA, _ip, _Store

STRANGER = np.asarray([-1.0, -1.0], dtype=np.float32)  # パパにもたいきにも負の似かた
CALL = "パジュ、いま何時？"


def _gate(ip, *, voice, source="voice", query=CALL) -> bool:
    trig = Trigger(kind="会話入力", query=query, source=source, voice=voice)
    return asyncio.run(ip._swallow_if_unheard(trig))


def _with_threshold(value=0.3, **kw):
    ip, a, s = _ip(**kw)
    a.config.recognition.voice_stranger_threshold = value
    a._nudge_drive = AsyncMock()
    a._someone_here = MagicMock(return_value=True)
    ip._load_silence = lambda: None  # type: ignore[method-assign]  # 黙っていての依頼は無い（DB を読まない）
    return ip, a, s


def test_the_threshold_is_unset_by_default():
    from familiar_agent.config import RecognitionConfig

    assert RecognitionConfig().voice_stranger_threshold is None


def test_without_a_threshold_a_stranger_is_not_judged():
    ip, a, s = _with_threshold(None)
    assert ip._voice_stranger(STRANGER) is False


def test_a_stranger_calling_the_name_is_dropped_before_the_window(caplog):
    ip, a, s = _with_threshold()
    with caplog.at_level(logging.INFO, logger="familiar_agent.loop.event_loop"):
        assert _gate(ip, voice=STRANGER) is True
    assert not ip._wake_window().is_open(asyncio.run(_now()))
    a._persons.set_active.assert_not_called()
    a._persons.reset_to_default.assert_not_called()
    assert s.added == []
    assert any("知らない声なので捨てた" in r.getMessage() for r in caplog.records)


async def _now() -> float:
    import time

    return time.monotonic()


def test_a_stranger_while_silenced_is_not_kept(monkeypatch):
    ip, a, s = _with_threshold()
    monkeypatch.setattr("familiar_agent.silence_state.is_silenced", lambda *a, **k: True)
    ip._load_silence = lambda: MagicMock(reason="")  # type: ignore[method-assign]
    ip._note_muted = AsyncMock()  # type: ignore[method-assign]
    assert _gate(ip, voice=STRANGER) is True
    ip._note_muted.assert_not_awaited()


@pytest.mark.parametrize(
    "voice,source",
    [(PAPA, "voice"), (None, "voice"), (STRANGER, "keyboard")],
    ids=["known", "no-voiceprint", "keyboard"],
)
def test_known_voices_no_voiceprint_and_keyboard_are_not_strangers(voice, source):
    ip, a, s = _with_threshold()
    assert ip._voice_stranger(voice if source == "voice" else None) is False
    _gate(ip, voice=voice, source=source)
    a._nudge_drive.assert_not_awaited()  # 窓が閉じていても押し上げは窓の外の道（いまどおり）だけ


def test_no_registered_voice_means_no_stranger():
    ip, a, s = _with_threshold(store=_Store())
    assert ip._voice_stranger(STRANGER) is False


def test_a_stranger_raises_safety_only_when_someone_is_here():
    ip, a, s = _with_threshold(1.5)  # 知っている声でも基準に届かない値でも、BOND には行かない
    _gate(ip, voice=PAPA)
    a._nudge_drive.assert_awaited_once()
    assert a._nudge_drive.await_args.args[0] == "safety"
    assert a._nudge_drive.await_args.args[1] == pytest.approx(0.10)

    ip, a, s = _with_threshold()
    a._someone_here = MagicMock(return_value=False)
    _gate(ip, voice=STRANGER)
    a._nudge_drive.assert_not_awaited()
