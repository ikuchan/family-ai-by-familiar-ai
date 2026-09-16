"""`/reload`——`.env` を読み直して常時集音を立て直す（2026-09-16 実機）。

`.env` は起動時に一度だけ読まれ（`bootstrap.py`）、マイクの倍率は集音開始時、`STT_*` の
値は STT セッション生成時に固定される。値を変えるたびに再起動していたので、GUI の
`/reload` で読み直す。元の `/reload` は `agent.reload_md_files` を呼んでいたが、その
メソッドは存在せず（grep 0 件）、押しても壊れるだけだった。

効くのは**集音の開始時に読まれるもの**だけ（マイク機器・`AUDIO_INPUT_GAIN`・`STT_*`・
担い手）。LLM・鍵・カメラ・声の担い手は起動時に組んだままなので、応答にその旨を添える。
"""

from __future__ import annotations

import asyncio
import os

from familiar_agent.env_reload import reload_env


def test_the_file_overrides_what_the_process_already_had(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("AUDIO_INPUT_GAIN=2.5\nSTT_MIN_SEGMENT_SEC=1.0\n", encoding="utf-8")
    monkeypatch.setenv("AUDIO_INPUT_GAIN", "2.0")

    report = reload_env(env)

    assert os.environ["AUDIO_INPUT_GAIN"] == "2.5"
    assert os.environ["STT_MIN_SEGMENT_SEC"] == "1.0"
    assert report.path == env


def test_the_summary_names_the_values_the_microphone_will_use(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("AUDIO_INPUT_GAIN=2.5\nSTT_MIN_SEGMENT_SEC=1.0\n", encoding="utf-8")
    monkeypatch.delenv("STT_VAD_SILENCE_SEC", raising=False)

    text = reload_env(env).summary()

    assert ".env を読み直した" in text
    assert "AUDIO_INPUT_GAIN=2.5" in text
    assert "STT_MIN_SEGMENT_SEC=1.0" in text
    assert "STT_VAD_SILENCE_SEC=（既定）" in text
    assert "再起動" in text  # 効かないものがあることを伝える


def test_a_missing_file_is_reported_not_raised(tmp_path):
    report = reload_env(tmp_path / "nothing.env")
    assert "見つからない" in report.summary()


# ── GUI の `/reload` ────────────────────────────────────────────────────────


class _Log:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def append_line(self, text: str) -> None:
        self.lines.append(text)


class _Controller:
    def __init__(self, label: str) -> None:
        self.label = label
        self.stopped = False
        self.started = False
        self.on_partial = self.on_committed = self.on_restart = None

    @property
    def engine_label(self) -> str:
        return self.label

    async def stop(self) -> None:
        self.stopped = True

    async def start(self, loop, queue) -> None:  # noqa: ANN001
        self.started = True


def _stub_window(old: _Controller):
    from familiar_agent.gui import FamiliarWindow

    win = FamiliarWindow.__new__(FamiliarWindow)
    win._closing = False
    win._log = _Log()
    win._input_queue = asyncio.Queue()
    win._realtime_stt = old
    win._realtime_stt_task = None
    win._agent = object()
    win._set_last_error = lambda _m: None
    win._restart_stt_btn = None
    return win


def test_reload_rereads_env_and_rebuilds_the_listener(tmp_path, monkeypatch):
    """古い集音を止め、新しい設定で作り直して起動する。"""
    from familiar_agent import gui as gui_mod
    from familiar_agent.gui import FamiliarWindow

    env = tmp_path / ".env"
    env.write_text("AUDIO_INPUT_GAIN=3.0\n", encoding="utf-8")
    monkeypatch.setenv("FAMILIAR_ENV_FILE", str(env))
    monkeypatch.setenv("AUDIO_INPUT_GAIN", "2.0")
    old = _Controller("faster-whisper")
    new = _Controller("faster-whisper")
    monkeypatch.setattr(gui_mod, "create_realtime_stt_controller", lambda: new)
    win = _stub_window(old)

    asyncio.run(FamiliarWindow._do_reload(win))

    assert os.environ["AUDIO_INPUT_GAIN"] == "3.0"
    assert old.stopped and new.started
    assert win._realtime_stt is new
    assert any("読み直した" in line for line in win._log.lines)
