"""声の担い手・声の中身・出口をログに残す（知-ak-に・2026-10-12・本人の決定）。

声が「今日は」だけで切れたとき（実機 11:41・11:49）、ログからは、合成した声の中身が切れていたのか、
鳴らす側が止まったのか、出口から先で止まったのかを分けられなかった。何で話しているかも出ておらず、
`.env` が担い手の決定（`根拠台帳` §45）を上書きしていたことに気づけなかった。
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from familiar_agent.tools import tts

RATE = 24000
#: 本物の `_sink_state`。`conftest` が試験ごとに差し替えるので、読み込んだときに取っておく。
_REAL_SINK_STATE = tts._sink_state


def _wav(tmp_path, *parts: "tuple[str, float]"):
    """("音", 秒)／("無音", 秒) を並べた WAV。"""
    chunks = []
    for kind, sec in parts:
        n = int(sec * RATE)
        t = np.arange(n) / RATE
        chunks.append(0.5 * np.sin(2 * np.pi * 220 * t) if kind == "音" else np.zeros(n))
    p = tmp_path / "v.wav"
    sf.write(str(p), np.concatenate(chunks).astype("float32"), RATE)
    return p


# ── 声の中身（A） ──────────────────────────────────────────────


def test_a_voice_that_goes_silent_shows_where_the_sound_ends(tmp_path):
    data, rate = sf.read(str(_wav(tmp_path, ("音", 1.0), ("無音", 3.0))))
    end, gap = tts._voice_shape(data, rate)
    assert end == pytest.approx(1.0, abs=0.03)
    assert gap == pytest.approx(3.0, abs=0.03)


def test_an_all_silent_voice_ends_at_zero(tmp_path):
    """**反証の側**：音が無ければ、有音の終わりは 0。"""
    data, rate = sf.read(str(_wav(tmp_path, ("無音", 2.0))))
    end, _ = tts._voice_shape(data, rate)
    assert end == 0.0


def test_an_unbroken_voice_has_no_gap(tmp_path):
    """**反証の側**：途切れの無い声は、最長の無音 0。"""
    data, rate = sf.read(str(_wav(tmp_path, ("音", 2.0))))
    end, gap = tts._voice_shape(data, rate)
    assert end == pytest.approx(2.0, abs=0.03)
    assert gap == 0.0


# ── 鳴らした行に、声の中身と出口を足す（A・B） ──────────────────


@pytest.fixture
def pw(monkeypatch):
    """偽の `pw-play`。すぐ戻り値 0 で終わる。"""
    monkeypatch.setattr(tts, "_pw_play_binary", lambda: "/usr/bin/pw-play")
    monkeypatch.setattr(tts, "_pw_target", lambda: None)

    class _Proc:
        returncode = 0

        def __init__(self, args, **kw):
            pass

        def communicate(self, timeout=None):
            time.sleep(0.05)
            return b"", b""

        def kill(self):
            pass

    real = tts.subprocess.Popen
    monkeypatch.setattr(
        tts.subprocess,
        "Popen",
        lambda args, **kw: (
            _Proc(args) if args and args[0] == "/usr/bin/pw-play" else real(args, **kw)
        ),
    )


def _played_line(caplog) -> str:
    return next(r.getMessage() for r in caplog.records if "声を鳴らした" in r.getMessage())


def test_the_play_line_shows_the_voice_shape_and_the_sink(pw, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(tts, "_sink_state", lambda: "Yamaha YVC-300 モノ・音量 0.68・ミュートなし")
    with caplog.at_level("INFO"):
        ok = asyncio.run(
            tts._play_via_sounddevice(str(_wav(tmp_path, ("音", 0.5), ("無音", 1.0))), gain=1.0)
        )
    assert ok is True
    line = _played_line(caplog)
    assert "有音の終わり 0.50 秒" in line and "最長の無音 1.00 秒" in line
    assert "既定の出口 Yamaha YVC-300 モノ・音量 0.68・ミュートなし" in line


def test_a_failed_sink_query_does_not_stop_the_voice(pw, tmp_path, monkeypatch, caplog):
    """**反証の側**：出口を聞けなくても声は鳴らし、行には「不明」と出す。"""

    def _boom():
        raise OSError("wpctl が無い")

    monkeypatch.setattr(tts, "_sink_state", _boom)
    with caplog.at_level("INFO"):
        ok = asyncio.run(tts._play_via_sounddevice(str(_wav(tmp_path, ("音", 0.2))), gain=1.0))
    assert ok is True
    assert "既定の出口 不明" in _played_line(caplog)


def test_the_sink_is_read_from_wpctl(monkeypatch):
    """`wpctl` の出力から、名前・音量・ミュートを読む（本物の wpctl は呼ばない）。"""
    outs = {
        "get-volume": "Volume: 0.68 [MUTED]\n",
        "inspect": '  * node.description = "Yamaha YVC-300 モノ"\n  * node.name = "alsa_output.usb"\n',
    }

    def _run(args, **kw):
        return SimpleNamespace(returncode=0, stdout=outs[args[1]])

    monkeypatch.setattr(tts.subprocess, "run", _run)
    assert _REAL_SINK_STATE() == "Yamaha YVC-300 モノ・音量 0.68・ミュートあり"


# ── 声ごとの担い手 ───────────────────────────────────────────


def _tool(engine: str, fallback: str = "") -> "tts.TTSTool":
    return tts.TTSTool("", "", engine=engine, fallback_engine=fallback)


def test_each_voice_says_which_engine_spoke(monkeypatch, caplog):
    t = _tool("sbv2")

    async def _say_sbv2(text, output):
        return f"Said: {text}"

    monkeypatch.setattr(t, "_say_sbv2", _say_sbv2)
    with caplog.at_level("INFO"):
        asyncio.run(t.say("今日は一日中晴れるみたいです", output="local"))
    assert "TTS: sbv2 で話した（14 字" in caplog.text


def test_a_fallback_shows_both_engines(monkeypatch, caplog):
    """**反証の側**：主が話せずに控えへ落ちたら、主の失敗と控えの成功が両方残る。"""
    t = _tool("gemini", "sbv2")

    async def _fail(text, output):
        return "話せなかった（合成に失敗）"

    async def _ok(text, output):
        return f"Said: {text}"

    monkeypatch.setattr(t, "_say_gemini", _fail)
    monkeypatch.setattr(t, "_say_sbv2", _ok)
    with caplog.at_level("INFO"):
        asyncio.run(t.say("こんにちは", output="local"))
    assert "TTS: gemini で話せなかった（5 字" in caplog.text
    assert "TTS: sbv2 で話した（5 字" in caplog.text


# ── 起動時の担い手 ───────────────────────────────────────────


def test_the_voice_setup_is_described():
    cfg = SimpleNamespace(
        engine="gemini",
        fallback_engine="sbv2",
        gemini_model="gemini-3.8-flash-lite-tts",
        gemini_voice="Charon",
        sbv2_model="jvnv-M2-jp",
        sbv2_style="Neutral",
        elevenlabs_model="eleven_flash_v2_5",
        output="local",
        gemini_api_key="秘密の鍵-gemini",
        elevenlabs_api_key="秘密の鍵-eleven",
    )
    line = tts.describe_voice(cfg)
    assert line == (
        "声の担い手：主 gemini（gemini-3.8-flash-lite-tts・Charon）"
        "／控え sbv2（jvnv-M2-jp・Neutral）・出力 local"
    )
    secret_g, secret_e = cfg.gemini_api_key, cfg.elevenlabs_api_key
    assert secret_g not in line and secret_e not in line


def test_no_fallback_is_said_plainly():
    """**反証の側**：控えが空なら「控え なし」。"""
    cfg = SimpleNamespace(
        engine="sbv2",
        fallback_engine="",
        sbv2_model="jvnv-M2-jp",
        sbv2_style="Neutral",
        output="local",
    )
    assert (
        tts.describe_voice(cfg)
        == "声の担い手：主 sbv2（jvnv-M2-jp・Neutral）／控え なし・出力 local"
    )
