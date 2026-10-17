"""パジュが終わるときに spotifyd も止め、起動時に残っていたら立ち上げ直す（知-bj・2026-10-17・本人の決定ア・イ）。

10/16 13:33 に立てた spotifyd が、パジュを `kill` で何度止めても残っていた。GUI は SIGINT しか受けず、SIGTERM では
後片付け（`agent.close()`）が走らない。次のパジュは「もう居る」と見て立てないので、自分のものとして止めることもない。

- ア：GUI が SIGTERM でも閉じる。閉じるときは同じ設定ファイルの spotifyd を止める。起動時に残っていれば立ち上げ直す。
- イ：`setpriv --pdeathsig TERM` で立ち上げ、パジュが強制終了しても一緒に終わる。印は立ち上げたスレッドの終わりでも
  効くので、立ち上げは終わらない専用のスレッド 1 本だけで行う。
"""

from __future__ import annotations

import asyncio
import signal
import threading
from unittest.mock import MagicMock

import pytest

from familiar_agent.io import spotifyd

CONF = "/home/u/.config/spotifyd/spotifyd.conf"


class _Proc:
    def __init__(self):
        self.terminated = False

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self):
        self.terminated = True


@pytest.fixture
def world(monkeypatch):
    w: dict = {"running": False, "spawned": [], "threads": [], "procs": {}, "killed": []}
    monkeypatch.setattr(spotifyd, "_started", None)
    monkeypatch.setattr(spotifyd, "_pgrep", lambda: w["running"] or bool(w["procs"]))
    monkeypatch.setattr(spotifyd, "find_binary", lambda: "/usr/bin/spotifyd")
    monkeypatch.setattr(spotifyd, "_setpriv", lambda: "/usr/bin/setpriv")
    monkeypatch.setattr(spotifyd, "_spotifyd_pids", lambda: list(w["procs"]))
    monkeypatch.setattr(spotifyd, "_cmdline", lambda pid: w["procs"].get(pid, []))
    monkeypatch.setattr(spotifyd, "_alive", lambda pid: pid in w["procs"])

    def kill(pid, sig):
        w["killed"].append(pid)
        w["procs"].pop(pid, None)

    monkeypatch.setattr(spotifyd.os, "kill", kill)

    def popen(args, **kw):
        w["spawned"].append(args)
        w["threads"].append(threading.current_thread())
        return _Proc()

    monkeypatch.setattr(spotifyd.subprocess, "Popen", popen)
    return w


# ── イ：親が死んだら一緒に終わる ──────────────────────────────────────────────


def test_it_is_started_under_setpriv_pdeathsig(world):
    assert spotifyd.ensure_running(CONF) == "started"
    (args,) = world["spawned"]
    assert args[:4] == ["/usr/bin/setpriv", "--pdeathsig", "TERM", "/usr/bin/spotifyd"]
    assert "--no-daemon" in args and CONF in args


def test_without_setpriv_it_starts_plainly(world, monkeypatch, caplog):
    """**反証の側**：`setpriv` が無い機体では、今までどおり直接立ち上げ、警告を残す。"""
    monkeypatch.setattr(spotifyd, "_setpriv", lambda: None)
    spotifyd.ensure_running(CONF)
    (args,) = world["spawned"]
    assert args[0] == "/usr/bin/spotifyd"
    assert "setpriv" in caplog.text


def test_it_is_always_started_from_the_one_owner_thread(world):
    """印は立ち上げたスレッドが終わっても効くので、終わらない専用のスレッドだけで立てる。"""

    async def twice():
        await spotifyd.ensure_running_async(CONF)
        spotifyd._started = None
        world["running"] = False
        await spotifyd.restart_async(CONF)

    asyncio.run(twice())
    first, second = world["threads"]
    assert first is second
    assert first is not threading.main_thread()
    assert first.name.startswith("spotifyd")


# ── ア：閉じるとき・起動するとき ──────────────────────────────────────────────


def test_closing_stops_ours_and_the_leftover_with_our_config(world):
    spotifyd.ensure_running(CONF)
    mine = spotifyd._started
    world["procs"] = {
        11: ["spotifyd", "--config-path", CONF],
        22: ["spotifyd", "--config-path", "/other.conf"],
    }
    spotifyd.stop_ours(CONF)
    assert mine.terminated and spotifyd._started is None
    assert world["killed"] == [11]  # **反証の側**：別の設定ファイルの spotifyd（22）には触らない


def test_startup_restarts_a_leftover(world, monkeypatch):
    world["procs"] = {11: ["spotifyd", "--config-path", CONF]}
    assert spotifyd.start_for_paju(CONF) == "started"
    assert world["killed"] == [11]
    assert len(world["spawned"]) == 1


def test_startup_without_a_leftover_just_starts(world):
    """**反証の側**：残っていなければ、止めずに立ち上げるだけ。"""
    assert spotifyd.start_for_paju(CONF) == "started"
    assert world["killed"] == []
    assert len(world["spawned"]) == 1


def test_the_agent_uses_the_new_ways_in_and_out():
    import inspect

    from familiar_agent import agent

    src = inspect.getsource(agent)
    assert "spotifyd.start_for_paju" in src
    assert "spotifyd.stop_ours(" in src


# ── ア-1：GUI が SIGTERM でも閉じる ───────────────────────────────────────────


def test_sigterm_quits_the_gui():
    from familiar_agent.gui import _install_quit_signals

    app = MagicMock()
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        _install_quit_signals(app)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        app.quit.assert_called_once()
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        assert app.quit.call_count == 2
    finally:
        for s, h in before.items():
            signal.signal(s, h)
