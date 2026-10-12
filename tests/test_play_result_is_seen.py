"""声を鳴らした結果を見る（知-ak-は・2026-10-12・本人の決定 案2）。

`pw-play` の戻り値とエラー出力を捨て、時間内に終われば「鳴った」とみなしていた。実機 11:41・11:49 で声が
「今日は」だけで切れたとき、アプリのログからは何が起きたか分からなかった。毎回、戻り値・鳴っていた秒数・声の
長さ・出口を残し、戻り値が 0 でなければ話せなかったとする。失敗したとき、同じ声をもう一度鳴らさない。
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest
import soundfile as sf

from familiar_agent.tools import tts


def _wav(tmp_path, seconds: float):
    p = tmp_path / "v.wav"
    sf.write(str(p), np.zeros(int(seconds * 24000), dtype="float32"), 24000)
    return p


@pytest.fixture
def pw(monkeypatch):
    """偽の `pw-play`。`code` で戻り値、`err` でエラー出力、`takes` で鳴っていた秒数を決める。"""
    seen: dict = {"calls": 0, "code": 0, "err": b"", "takes": 0.0}
    monkeypatch.setattr(tts, "_pw_play_binary", lambda: "/usr/bin/pw-play")
    monkeypatch.setattr(tts, "_pw_target", lambda: None)

    class _Proc:
        def __init__(self, args, **kw):
            seen["calls"] += 1
            self.args = args
            self.returncode = None

        def communicate(self, timeout=None):
            time.sleep(seen["takes"])
            self.returncode = seen["code"]
            return b"", seen["err"]

        def wait(self, timeout=None):
            return self.communicate(timeout)[0] or self.returncode

        def kill(self):
            pass

    real = tts.subprocess.Popen

    def popen(args, **kw):
        # pw-play の呼び出しだけを偽物に（ライブラリの読み込みが使う Popen はそのまま）。
        return _Proc(args, **kw) if args and args[0] == "/usr/bin/pw-play" else real(args, **kw)

    monkeypatch.setattr(tts.subprocess, "Popen", popen)
    return seen


def test_a_failed_play_is_not_counted_as_spoken(pw, tmp_path, caplog):
    pw["code"], pw["err"] = 1, b"error: target not found\n"
    with caplog.at_level("WARNING"):
        ok = asyncio.run(tts._play_via_sounddevice(str(_wav(tmp_path, 0.2)), gain=1.0))
    assert ok is False
    assert "戻り値 1" in caplog.text and "target not found" in caplog.text


def test_a_finished_play_leaves_one_line(pw, tmp_path, caplog):
    pw["takes"] = 0.3  # 声（0.2 秒）より長くかかった＝最後まで鳴った
    with caplog.at_level("INFO"):
        ok = asyncio.run(tts._play_via_sounddevice(str(_wav(tmp_path, 0.2)), gain=1.0))
    assert ok is True
    line = next(r.getMessage() for r in caplog.records if "声を鳴らした" in r.getMessage())
    assert "戻り値 0" in line and "声 0.20 秒" in line and "出口 既定" in line
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_a_play_that_ends_before_the_voice_is_warned_but_counted(pw, tmp_path, caplog):
    """**反証の側**：早く終わっても戻り値が 0 なら話したことにする（警告だけ残す）。"""
    with caplog.at_level("WARNING"):
        ok = asyncio.run(tts._play_via_sounddevice(str(_wav(tmp_path, 1.0)), gain=1.0))
    assert ok is True
    assert "声の長さより早く終わった" in caplog.text


def test_a_failed_play_is_not_played_again(pw, tmp_path, monkeypatch):
    """`_play_local` は最初に pw-play の道を試し、失敗すると最後にもう一度同じ道を呼んでいた（同じ声が 2 回鳴る）。"""
    monkeypatch.setenv("AUDIO_INPUT_DEVICE", "Yamaha")
    monkeypatch.setattr(
        tts.shutil, "which", lambda name: None
    )  # paplay・mpv が無い機体（いまの実機）
    pw["code"] = 1
    ok = asyncio.run(tts._play_local(str(_wav(tmp_path, 0.2))))
    assert ok is False
    assert pw["calls"] == 1
