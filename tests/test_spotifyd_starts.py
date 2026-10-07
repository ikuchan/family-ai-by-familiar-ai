"""パジュが spotifyd を立ち上げる（2026-10-07 実機・本人の決定）。

実機 2026-10-07 12:59、「ケイマンをランダムでかけて」で音楽が鳴らなかった。spotifyd（Spotify の音をこの機体で
鳴らす裏方）は手で起動する決まりで、その日は立っていなかった。

- 起動時：音楽の器があって spotifyd が居なければ立ち上げる（待たない）。
- `play_music` の前：居なければ立ち上げ、機器「パジュ」が Spotify に見えるまで 1 秒おきに最大 10 秒〔仮〕待つ。
- 終了時：自分で立てたものだけを止める。人が立てたものには触らない。
- 居るかはプロセスで見る（MPRIS の口は機器が現役になるまで出ないので、待機中は見えない）。
- 本体は PATH → `~/.local/bin/spotifyd` の順に探す。見つからなければ警告を残して落ちない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from familiar_agent.io import spotifyd


@pytest.fixture(autouse=True)
def _forget(monkeypatch):
    monkeypatch.setattr(spotifyd, "_started", None)
    yield


class _Proc:
    def __init__(self):
        self.terminated = False

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self):
        self.terminated = True


def _world(monkeypatch, *, running=False, binary="/usr/bin/spotifyd"):
    w = {"running": running, "spawned": []}
    monkeypatch.setattr(spotifyd, "_pgrep", lambda: w["running"])
    monkeypatch.setattr(spotifyd, "find_binary", lambda: binary)

    def popen(args, **kw):
        w["spawned"].append(args)
        return _Proc()

    monkeypatch.setattr(spotifyd.subprocess, "Popen", popen)
    return w


# ── 立ち上げる・立ち上げない ───────────────────────────────────────────────────


def test_it_starts_when_not_running(monkeypatch):
    w = _world(monkeypatch)
    assert spotifyd.ensure_running("~/.config/spotifyd/spotifyd.conf") == "started"
    (args,) = w["spawned"]
    assert args[0] == "/usr/bin/spotifyd" and "--no-daemon" in args
    assert args[args.index("--config-path") + 1].endswith(".config/spotifyd/spotifyd.conf")
    assert "~" not in args[args.index("--config-path") + 1]


def test_it_does_not_start_twice(monkeypatch):
    w = _world(monkeypatch, running=True)
    assert spotifyd.ensure_running("c") == "running"
    assert w["spawned"] == []


def test_our_own_live_process_counts_as_running(monkeypatch):
    w = _world(monkeypatch)
    spotifyd.ensure_running("c")
    assert spotifyd.ensure_running("c") == "running"
    assert len(w["spawned"]) == 1


def test_without_the_binary_it_warns_and_does_not_fall(monkeypatch, caplog):
    w = _world(monkeypatch, binary=None)
    assert spotifyd.ensure_running("c") == "missing"
    assert w["spawned"] == []
    assert "spotifyd" in caplog.text


def test_the_binary_is_looked_for_on_path_then_local_bin(monkeypatch, tmp_path):
    monkeypatch.setattr(spotifyd.shutil, "which", lambda name: "/opt/bin/spotifyd")
    assert spotifyd.find_binary() == "/opt/bin/spotifyd"
    local = tmp_path / ".local" / "bin" / "spotifyd"
    local.parent.mkdir(parents=True)
    local.write_text("")
    monkeypatch.setattr(spotifyd.shutil, "which", lambda name: None)
    monkeypatch.setattr(spotifyd.Path, "home", lambda: tmp_path)
    assert spotifyd.find_binary() == str(local)
    local.unlink()
    assert spotifyd.find_binary() is None


# ── 終了時 ──────────────────────────────────────────────────────────────────


def test_only_the_one_we_started_is_stopped(monkeypatch):
    _world(monkeypatch)
    spotifyd.ensure_running("c")
    proc = spotifyd._started
    spotifyd.stop_started()
    assert proc.terminated and spotifyd._started is None


def test_one_someone_else_started_is_left_alone(monkeypatch):
    _world(monkeypatch, running=True)
    spotifyd.ensure_running("c")
    spotifyd.stop_started()  # 何も持っていないので何もしない（落ちない）


# ── 機器が見えるまで待つ ───────────────────────────────────────────────────────


def _waits(seen_after: "int | None", *, wait_sec=10.0):
    web = MagicMock()
    calls = {"n": 0}

    def device_id(name):
        calls["n"] += 1
        return "dev" if seen_after is not None and calls["n"] > seen_after else None

    web.device_id = device_id
    slept: list[float] = []

    async def sleep(s):
        slept.append(s)

    got = asyncio.run(spotifyd.wait_for_device(web, "パジュ", wait_sec=wait_sec, sleep=sleep))
    return got, slept


def test_it_waits_until_the_device_is_seen():
    got, slept = _waits(seen_after=3)
    assert got is True and slept == [1.0, 1.0, 1.0]


def test_it_gives_up_after_wait_sec():
    got, slept = _waits(seen_after=None)
    assert got is False and sum(slept) == pytest.approx(10.0)


# ── 鳴らす前に確かめる ──────────────────────────────────────────────────────────


def _tool(monkeypatch, state: str):
    from unittest.mock import AsyncMock

    from familiar_agent.tools.music import MusicTool

    order: list[str] = []
    monkeypatch.setattr(
        spotifyd, "ensure_running", lambda conf: order.append(f"ensure:{conf}") or state
    )

    async def wait(web, name, *, wait_sec, sleep=asyncio.sleep):
        order.append(f"wait:{name}:{wait_sec}")
        return True

    monkeypatch.setattr(spotifyd, "wait_for_device", wait)
    io = MagicMock()
    io.play = AsyncMock(side_effect=lambda bus, uri: order.append("play") or True)
    io.set_shuffle = AsyncMock()
    web = MagicMock()
    web.activate = MagicMock(side_effect=lambda name: order.append("activate") or True)
    tool = MusicTool(
        io=io,
        bus=lambda: MagicMock(),
        table=lambda: (("ケイマン", "spotify:playlist:x", True),),
        web=web,
        device_name="パジュ",
        spotifyd_conf="conf",
        spotifyd_wait_sec=10.0,
    )
    return tool, order


def test_play_starts_spotifyd_and_waits_for_the_device_first(monkeypatch):
    tool, order = _tool(monkeypatch, "started")
    text, ok = asyncio.run(tool._play({"name": "ケイマン"}))
    assert ok
    assert order == ["ensure:conf", "wait:パジュ:10.0", "activate", "play"]


def test_play_does_not_wait_when_it_was_already_running(monkeypatch):
    tool, order = _tool(monkeypatch, "running")
    asyncio.run(tool._play({"name": "ケイマン"}))
    assert order == ["ensure:conf", "activate", "play"]


# ── 起動と終了の結線 ────────────────────────────────────────────────────────────


def test_config_defaults(monkeypatch):
    from familiar_agent.config import AgentConfig

    monkeypatch.delenv("MUSIC_SPOTIFYD_WAIT_SEC", raising=False)
    monkeypatch.delenv("SPOTIFYD_CONFIG", raising=False)
    cfg = AgentConfig()
    assert cfg.spotifyd_wait_sec == 10.0
    assert cfg.spotifyd_config == "~/.config/spotifyd/spotifyd.conf"


def test_the_agent_starts_it_at_startup_and_stops_it_at_exit():
    import inspect

    from familiar_agent import agent

    src = inspect.getsource(agent)
    assert "spotifyd.ensure_running" in src
    assert "spotifyd.stop_started()" in src
    assert "spotifyd_conf=" in src and "spotifyd_wait_sec=" in src
