"""spotifyd を立ち上げる口（2026-10-07・本人の決定）。

spotifyd は Spotify の音をこの機体で鳴らす裏方で、以前は手で起動する決まりだった。実機 2026-10-07 12:59、
立っていなかったので「ケイマンをかけて」で鳴らなかった。起動時と `play_music` の前に、居なければ立ち上げる。

- **居るかはプロセスで見る。** MPRIS の口は機器が現役になってから出る（`io/music`）ので、待機中は見えない。
  自分で立てたものはその手綱で、ほかは `pgrep -x spotifyd` で見る。
- **自分で立てたものだけを止める**（SBV2 の合成サーバーと同じ扱い）。人が立てたものには触らない。
- 本体が見つからなければ警告を残すだけで落ちない（音楽は「かけられなかった」と返る）。
- **音の出口はこの機体**（2026-10-07 実機）。アプリを遠隔デスクトップのターミナルから起動すると、その接続用の音の口を
  指す環境変数を引き継ぎ、spotifyd の音が遠隔デスクトップへ流れて Yamaha から鳴らなかった。立ち上げるときだけ外し、
  この機体の音の口（`<XDG_RUNTIME_DIR>/pulse/native`・既定の出口が Yamaha）を指す（`_audio_env`）。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

_started: "subprocess.Popen | None" = None


def find_binary() -> "str | None":
    """spotifyd の本体。PATH → `~/.local/bin/spotifyd` の順に探す。無ければ None。"""
    found = shutil.which("spotifyd")
    if found:
        return found
    local = Path.home() / ".local" / "bin" / "spotifyd"
    return str(local) if local.exists() else None


def _audio_env() -> "dict[str, str]":
    """spotifyd に渡す環境。遠隔デスクトップの音の口を外し、この機体の音の口を指す（`io/audio_env`）。"""
    from .audio_env import local_audio_env

    return local_audio_env(os.environ)


def _pgrep() -> bool:
    """自分の外で spotifyd が動いているか（差し替え点）。"""
    try:
        return subprocess.run(["pgrep", "-x", "spotifyd"], capture_output=True).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def is_running() -> bool:
    """spotifyd が居るか。自分で立てたものが生きていれば真、そうでなければプロセスを探す。"""
    if _started is not None and _started.poll() is None:
        return True
    return _pgrep()


def ensure_running(config_path: str) -> str:
    """居なければ立ち上げる（待たない）。`running`（もう居た）・`started`（立てた）・`missing`（立てられない）。"""
    global _started
    if is_running():
        return "running"
    binary = find_binary()
    if binary is None:
        logger.warning("spotifyd が見つからないので立ち上げられない（音楽は鳴らない）")
        return "missing"
    conf = os.path.expanduser(config_path)
    try:
        _started = subprocess.Popen(
            [binary, "--no-daemon", "--config-path", conf],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_audio_env(),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("spotifyd を立ち上げられなかった: %s", e)
        return "missing"
    logger.info("spotifyd を立ち上げた（%s）", conf)
    return "started"


#: 立ち上げ直すとき、止めた spotifyd が終わるのを待つ最大の秒数（〔仮〕・2026-10-13 本人の決定）。
STOP_WAIT_SEC = 5.0


def _spotifyd_pids() -> "list[int]":
    """動いている spotifyd の pid（差し替え点）。"""
    try:
        out = subprocess.run(["pgrep", "-x", "spotifyd"], capture_output=True, text=True).stdout
    except Exception:  # noqa: BLE001
        return []
    return [int(x) for x in out.split() if x.isdigit()]


def _cmdline(pid: int) -> "list[str]":
    """プロセスの起動の引数（差し替え点）。読めなければ空。"""
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")
    except Exception:  # noqa: BLE001
        return []


def _alive(pid: int) -> bool:
    """プロセスがまだ居るか（差し替え点）。"""
    return Path(f"/proc/{pid}").exists()


def restart(config_path: str, *, sleep: "Callable[[float], Any] | None" = None) -> str:
    """同じ設定ファイルで動いている spotifyd を止めて、立ち上げ直す（知-bd・2026-10-13・本人の決定ア）。

    動いていても Spotify から機器が見えなくなることがある（10/13 19:58・20:12、10/07 から動き続けていた spotifyd）。
    アプリを起動し直すと自分で立てたかが分からなくなるので、誰が立てたかではなく、起動の引数にパジュと同じ設定ファイルを
    含むかで決める。別の設定ファイルで動いている spotifyd には触らない。終わるのを最大 `STOP_WAIT_SEC` 秒待つ。
    """
    import signal
    import time

    global _started
    pause = sleep or time.sleep
    conf = os.path.expanduser(config_path)
    ours = [pid for pid in _spotifyd_pids() if conf in _cmdline(pid)]
    for pid in ours:
        with contextlib.suppress(Exception):
            os.kill(pid, signal.SIGTERM)
    waited = 0.0
    while any(_alive(pid) for pid in ours) and waited < STOP_WAIT_SEC:
        pause(0.2)
        waited += 0.2
    _started = None
    logger.info("spotifyd を立ち上げ直す（止めた %d 件）", len(ours))
    return ensure_running(config_path)


async def wait_for_device(
    web: Any,
    name: str,
    *,
    wait_sec: float,
    sleep: "Callable[[float], Awaitable[Any]]" = asyncio.sleep,
) -> bool:
    """立ち上げた直後、機器 `name` が Spotify に見えるまで 1 秒おきに見る。`wait_sec` たったら諦める。"""
    waited = 0.0
    while True:
        with contextlib.suppress(Exception):
            if await asyncio.to_thread(web.device_id, name):
                return True
        if waited >= wait_sec:
            logger.info(
                "音楽：機器「%s」が %.0f 秒たっても見えない（そのまま進む）", name, wait_sec
            )
            return False
        await sleep(1.0)
        waited += 1.0


def stop_started() -> None:
    """自分で立てた spotifyd だけを止める。持っていなければ何もしない。"""
    global _started
    proc = _started
    _started = None
    if proc is None or proc.poll() is not None:
        return
    logger.info("spotifyd を止める（自分で立てたもの）")
    with contextlib.suppress(Exception):
        proc.terminate()
