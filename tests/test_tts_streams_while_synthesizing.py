"""Gemini の声は、合成しながら鳴らす（出-by・2026-10-13・本人の決定）。

10/12 23:59 の答え（73 字・声 13.48 秒）は、最初の音が 0.91 秒で届いたのに、全部そろう 3.62 秒まで鳴らさなかった。
最初の生の音が届いたら `pw-play --raw … -` を起こし、届いた音を順に流し込む。音が届かないまま 10 秒〔仮〕たったら止める。
1 つも鳴らしていなければ控え（SBV2）へ落ちるが、途中まで鳴らしたあとに失敗したら控えには落ちない（同じ文を頭から
2 回聞かせない）。WAV で届いた・カメラのスピーカー・`pw-play` が無い、ときは今までどおり溜めてから鳴らす。
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from familiar_agent.tools import tts
from tests.test_tts_gemini import _chunk, _tool

PCM = b"\x10\x00" * 2400  # 0.1 秒ぶん（24 kHz・16 bit・モノ）


class _Pipe:
    def __init__(self, log):
        self.log = log

    def write(self, data):
        self.log.append((time.monotonic(), bytes(data)))

    def flush(self):
        pass

    def close(self):
        self.log.append((time.monotonic(), None))


@pytest.fixture
def pw(monkeypatch):
    """偽の `pw-play`。標準入力に書かれた音と、起こされた時刻を残す。"""
    seen: dict = {"args": None, "started": None, "writes": [], "code": 0, "killed": False}
    monkeypatch.setattr(tts, "_pw_play_binary", lambda: "/usr/bin/pw-play")
    monkeypatch.setattr(tts, "_pw_target", lambda: None)

    class _Proc:
        def __init__(self, args, **kw):
            seen["args"], seen["started"] = args, time.monotonic()
            self.stdin = _Pipe(seen["writes"])
            self.stderr = MagicMock(read=lambda: b"")
            self.returncode = None

        def wait(self, timeout=None):
            self.returncode = seen["code"]
            return self.returncode

        def kill(self):
            seen["killed"] = True
            self.returncode = -9

    real = tts.subprocess.Popen

    class _Popen:
        """`subprocess.Popen` の代わり。型として添字を付けるライブラリがあるので、関数ではなくクラスにする。"""

        def __new__(cls, args, **kw):
            if args and args[0] == "/usr/bin/pw-play":
                return _Proc(args, **kw)
            return real(args, **kw)

        def __class_getitem__(cls, item):
            return cls

    monkeypatch.setattr(tts.subprocess, "Popen", _Popen)
    return seen


def _slow_chunks(n=4, gap=0.15, mime="audio/pcm;rate=24000", ended=None):
    for i in range(n):
        if i:
            time.sleep(gap)
        yield _chunk(PCM, mime)
    if ended is not None:
        ended.append(time.monotonic())


def _say(tool, chunks, *, played=None):
    client = MagicMock()
    client.models.generate_content_stream.return_value = chunks

    async def fake_play(path, gain=1.0):
        if played is not None:
            played.append(path)
        return True

    with (
        patch("familiar_agent.tools.tts._gemini_client", return_value=client),
        patch("familiar_agent.tools.tts._play_local", new=fake_play),
    ):
        return asyncio.run(tool.say("明日は晴れです"))


def test_playing_starts_before_the_synthesis_ends(pw, caplog):
    ended: list = []
    with caplog.at_level("INFO"):
        result = _say(_tool(), _slow_chunks(ended=ended))
    assert result.startswith("Said:")
    assert pw["started"] < ended[0] - 0.2  # 最後の音が届く前に鳴らし始めた
    sent = [d for _t, d in pw["writes"] if d is not None]
    assert b"".join(sent) == PCM * 4  # 届いた順にすべて流した
    assert pw["writes"][-1][1] is None  # 届き終わったら閉じた
    args = pw["args"]
    assert "--raw" in args and "--rate=24000" in args and "--channels=1" in args
    assert "--format=s16" in args and args[-1] == "-"
    assert "鳴り始め" in caplog.text
    line = next(r.getMessage() for r in caplog.records if "声を鳴らした" in r.getMessage())
    assert "戻り値 0" in line and "有音の終わり" in line and "逐次" in line


def test_a_wav_reply_is_played_after_it_is_complete(pw):
    """**反証の側**：WAV で届いたら、今までどおり溜めてから鳴らす。"""
    played: list = []
    wav = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 32
    result = _say(_tool(), iter([_chunk(wav, "audio/wav")]), played=played)
    assert result.startswith("Said:")
    assert pw["started"] is None and played


def test_the_camera_speaker_is_not_streamed(pw):
    """**反証の側**：カメラのスピーカー（remote）は溜めてから。"""
    played: list = []
    tool = _tool(output="both")
    with patch("familiar_agent.tools.tts._play_via_go2rtc", return_value=(False, "no camera")):
        _say(tool, iter([_chunk(PCM, "audio/pcm;rate=24000")]), played=played)
    assert pw["started"] is None and played


def test_a_stalled_synthesis_is_stopped_without_falling_back(pw, monkeypatch, caplog):
    """途中まで鳴らしたあと音が届かなくなったら止める。控えへは落ちない（同じ文を 2 回聞かせない）。"""
    monkeypatch.setattr(tts, "_STREAM_IDLE_SEC", 0.2)
    stop = threading.Event()

    def stalled():
        yield _chunk(PCM, "audio/pcm;rate=24000")
        stop.wait(2.0)  # 次の音が届かない

    tool = _tool()
    tool.fallback_engine = "sbv2"
    sbv2 = MagicMock()

    async def _sbv2(text, output):
        sbv2(text)
        return "Said: 控え"

    tool._say_sbv2 = _sbv2  # type: ignore[method-assign]
    with caplog.at_level("WARNING"):
        result = _say(tool, stalled())
    stop.set()
    assert pw["killed"] is True
    assert result.startswith("Said:")
    sbv2.assert_not_called()
    assert "音が届かない" in caplog.text


def test_nothing_played_falls_back(pw):
    """**反証の側**：1 つも鳴らさないうちに失敗したら、控えへ落ちる。"""

    def broken():
        raise RuntimeError("接続が切れた")
        yield  # pragma: no cover

    tool = _tool()
    tool.fallback_engine = "sbv2"

    async def _sbv2(text, output):
        return "Said: 控え"

    tool._say_sbv2 = _sbv2  # type: ignore[method-assign]
    result = _say(tool, broken())
    assert result == "Said: 控え"
    assert pw["started"] is None


def test_a_failed_player_after_playing_does_not_fall_back(pw, caplog):
    pw["code"] = 1
    tool = _tool()
    tool.fallback_engine = "sbv2"
    sbv2 = MagicMock()

    async def _sbv2(text, output):
        sbv2(text)
        return "Said: 控え"

    tool._say_sbv2 = _sbv2  # type: ignore[method-assign]
    with caplog.at_level("WARNING"):
        result = _say(tool, _slow_chunks(n=2, gap=0.0))
    assert result.startswith("Said:")
    sbv2.assert_not_called()
    assert "戻り値 1" in caplog.text
