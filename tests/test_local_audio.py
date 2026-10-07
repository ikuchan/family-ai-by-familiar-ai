"""音の口はこの機体（知-ak-ろ 段 1・2026-10-07 実機）。

アプリを遠隔デスクトップ（Chrome Remote Desktop）のターミナルから起動すると、その接続用の音の口を指す環境変数を
引き継ぐ。spotifyd は段 4 で外したが、アプリ自身の声（PortAudio）も同じ環境を持つので、PipeWire 側の Yamaha の出口が
見えず、直に開く `hw:1,0` しか選べない。音楽が PipeWire で Yamaha を握っているあいだ、声は `Device unavailable` で
10 回消えた。起動の最初（sounddevice を読む前）に、プロセスの環境もこの機体の音の口へ向ける。
"""

from __future__ import annotations

import inspect

from familiar_agent.io import audio_env

_CRD = {
    "PULSE_RUNTIME_PATH": "/run/user/1000/crd_audio#K7euY9wjcA",
    "PULSE_SINK": "chrome_remote_desktop_session",
    "PIPEWIRE_REMOTE": "crd_audio#K7euY9wjcA/pipewire",
}


def _socket(tmp_path):
    (tmp_path / "pulse").mkdir()
    (tmp_path / "pulse" / "native").write_text("")
    return tmp_path


def test_the_remote_desktop_audio_is_replaced_by_the_local_one(tmp_path):
    run = _socket(tmp_path)
    env = {**_CRD, "XDG_RUNTIME_DIR": str(run), "HOME": "/home/x"}
    got = audio_env.local_audio_env(env)
    assert [k for k in _CRD if k in got] == []
    assert got.get("PULSE_SERVER") == f"unix:{run}/pulse/native"
    assert got["HOME"] == "/home/x"
    assert env["PULSE_SINK"] == "chrome_remote_desktop_session"  # 渡した辞書は変えない


def test_without_the_local_socket_it_only_drops(tmp_path, caplog):
    got = audio_env.local_audio_env({**_CRD, "XDG_RUNTIME_DIR": str(tmp_path)})
    assert [k for k in _CRD if k in got] == [] and got.get("PULSE_SERVER") is None
    assert "音の口" in caplog.text


def test_apply_rewrites_the_process_env(monkeypatch, tmp_path):
    run = _socket(tmp_path)
    for k, v in _CRD.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(run))
    monkeypatch.setenv("PULSE_SERVER", "")  # 後片付けで消えるよう、いったん置いてから消す
    monkeypatch.delenv("PULSE_SERVER")
    import os

    assert audio_env.apply_local_audio() is True
    assert [k for k in _CRD if k in os.environ] == []
    assert os.environ.get("PULSE_SERVER") == f"unix:{run}/pulse/native"


def test_apply_does_nothing_outside_a_remote_desktop(monkeypatch, tmp_path):
    for k in _CRD:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PULSE_SERVER", "")
    monkeypatch.delenv("PULSE_SERVER")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(_socket(tmp_path)))
    import os

    assert audio_env.apply_local_audio() is False
    assert os.environ.get("PULSE_SERVER") is None


def test_main_applies_it_before_anything_opens_audio():
    from familiar_agent import main

    src = inspect.getsource(main.main)
    assert "apply_local_audio()" in src
    assert src.index("setup_logging(") < src.index("apply_local_audio()")
    assert src.index("apply_local_audio()") < src.index(
        'if len(sys.argv) > 1 and sys.argv[1] == "mcp"'
    )


def test_spotifyd_uses_the_same_rule():
    from familiar_agent.io import spotifyd

    assert "local_audio_env" in inspect.getsource(spotifyd._audio_env)
