"""機器の見張り（知-bv・`設計方針_機器の見張り`）。スピーカー、マイク、カメラを間をおいて確かめ、使えなければ直す。

2026-11-14 の再起動のあと、YAMAHA YVC-300 の使い方が `off` になって声が内部オーディオへ出、カメラは起動の 18 秒後に
開けないまま開き直さなかった。どちらもパジュは気づかなかった。ここでは、直したことを INFO に、直せなかったことだけを
WARNING に残す。

音の命令（`wpctl`、`pw-dump`）はすべて `run_cmd` の口から呼ぶ。試験は偽の口に差し替え、本物の機器に触らない。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import subprocess
import time
from typing import Any, Awaitable, Callable, Protocol, Sequence

from ..config import DeviceWatchConfig
from ..core.device_watch import CheckResult, next_interval

logger = logging.getLogger(__name__)

RunCmd = Callable[[list[str]], str]

_PROFILE = "pro-audio"
_SINK_TRIES = 5  # 使い方を変えたあと、出口が現れるのを 1 秒おきに待つ回数


class Item(Protocol):
    """見張る機器 1 つ。`check` は使えるか（確かめないときは None）、`repair` は直し方（同期でも非同期でもよい）。"""

    name: str

    def check(self) -> "bool | None": ...

    def repair(self) -> Any: ...


def run_cmd(cmd: list[str]) -> str:
    """本物の命令を 5 秒の締切で回し、標準出力を返す。音の口はパジュが起動のときにこの機体へ向けてある（`io/audio_env`）。"""
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
    return r.stdout or ""


# ── スピーカー（§3.1）──────────────────────────────────────────────────────


def speaker_ok(name: str, run: RunCmd) -> bool:
    """既定の出口の説明に、機器の名前が入っているか。"""
    for line in run(["wpctl", "inspect", "@DEFAULT_AUDIO_SINK@"]).splitlines():
        if "node.description" in line and "=" in line:
            return name.lower() in line.split("=", 1)[1].lower()
    return False


def _objects(run: RunCmd) -> list[dict[str, Any]]:
    data = json.loads(run(["pw-dump"]) or "[]")
    return data if isinstance(data, list) else []


def _props(o: dict[str, Any]) -> dict[str, Any]:
    return (o.get("info") or {}).get("props") or {}


def _device(objs: list[dict[str, Any]], name: str) -> "dict[str, Any] | None":
    for o in objs:
        if (
            o.get("type") == "PipeWire:Interface:Device"
            and name.lower() in str(_props(o).get("device.description", "")).lower()
        ):
            return o
    return None


def _sink_id(objs: list[dict[str, Any]], name: str) -> "int | None":
    for o in objs:
        p = _props(o)
        if (
            p.get("media.class") == "Audio/Sink"
            and name.lower() in str(p.get("node.description", "")).lower()
        ):
            return int(o["id"])
    return None


def repair_speaker(name: str, run: RunCmd, *, sleep: Callable[[float], None] = time.sleep) -> bool:
    """使い方を `pro-audio` にし、出口が現れたら既定の出口にする。番号は毎回引く（起動ごとに変わる）。"""
    objs = _objects(run)
    dev = _device(objs, name)
    if dev is None:
        logger.info("スピーカーを直せない：機器が見つからない")
        return False
    params = (dev.get("info") or {}).get("params") or {}
    current = [p.get("name") for p in params.get("Profile") or []]
    if _PROFILE not in current:
        index = next(
            (p.get("index") for p in params.get("EnumProfile") or [] if p.get("name") == _PROFILE),
            None,
        )
        if index is None:
            logger.info("スピーカーを直せない：使い方 %s が選べない", _PROFILE)
            return False
        run(["wpctl", "set-profile", str(dev["id"]), str(index)])
        logger.info("スピーカーの使い方を %s にした（機器 %s）", _PROFILE, dev["id"])
        # 使い方を変えると、前の使い方の出口は消える。番号は変えたあとの一覧から引く。
        objs = []
    sink = _sink_id(objs, name)
    for _ in range(_SINK_TRIES):
        if sink is not None:
            break
        sleep(1.0)
        sink = _sink_id(_objects(run), name)
    if sink is None:
        logger.info("スピーカーを直せない：出口が現れない")
        return False
    run(["wpctl", "set-default", str(sink)])
    logger.info("既定の出口をスピーカー（出口 %d）にした", sink)
    return True


class Speaker:
    """スピーカー：既定の出口が機器か。名前の設定が無ければ確かめない。名前は毎回引く（設定の読み直しに付いていく）。"""

    name = "スピーカー"

    def __init__(self, device: Callable[[], str], run: RunCmd = run_cmd) -> None:
        self._device = device
        self._run = run

    def check(self) -> "bool | None":
        name = self._device()
        return speaker_ok(name, self._run) if name else None

    async def repair(self) -> None:
        await asyncio.to_thread(repair_speaker, self._device(), self._run)


class Fresh:
    """最後に受けた時刻で確かめる機器（マイク、カメラ）。`age` は最後に受けてからの秒（確かめないときは None）。

    `repair` は直す手を打ったか（False なら打っていない）を返してよい。打ったときは、新しい映像や音が来るまで
    `settle_sec` 秒まで待つ（開くのに時間がかかる）。来なければ、確かめ直しで古いままなので、直せなかったと分かる。
    待つのは見張りの仕事の中なので、取り消せばすぐ止まる。
    """

    def __init__(
        self,
        name: str,
        age: Callable[[], "float | None"],
        repair: Callable[[], Any],
        *,
        stale_sec: float,
        settle_sec: float,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.name = name
        self._age = age
        self._repair = repair
        self._stale = stale_sec
        self._settle = settle_sec
        self._sleep = sleep

    def check(self) -> "bool | None":
        a = self._age()
        if a is None:
            return None
        return a <= self._stale

    async def repair(self) -> None:
        out = self._repair()
        if inspect.isawaitable(out):
            out = await out
        if out is False:
            return
        waited = 0.0
        while waited < self._settle and self.check() is False:
            await self._sleep(1.0)
            waited += 1.0


def build_items(
    *,
    device_name: Callable[[], str],
    mic_age: Callable[[], "float | None"],
    mic_restart: Callable[[], Any],
    camera: Any,
    cfg: DeviceWatchConfig,
) -> list[Item]:
    """見張る機器を並べる。カメラはカメラの設定があるときだけ。

    スピーカーとマイクはいつも並べる（機器の名前の設定が無い、聞き取りを止めているときは、確かめの中で None になる）。
    """
    items: list[Item] = [
        Speaker(device_name),
        Fresh("マイク", mic_age, mic_restart, stale_sec=cfg.stale_sec, settle_sec=cfg.settle_sec),
    ]
    if camera is not None:
        items.append(
            Fresh(
                "カメラ",
                camera.frame_age,
                camera.reopen,
                stale_sec=cfg.stale_sec,
                settle_sec=cfg.settle_sec,
            )
        )
    return items


# ── 見張りの 1 回と、見張りの仕事 ──────────────────────────────────────────────


async def _check(item: Item) -> "bool | None":
    return await asyncio.to_thread(item.check)


async def check_once(items: Sequence[Item], *, retry_sec: float = 300.0) -> CheckResult:
    """どれも確かめ、使えないものを直して確かめ直す。1 つが落ちても、ほかは続ける。"""
    res = CheckResult()
    for item in items:
        try:
            ok = await _check(item)
            if ok is None or ok:
                continue
            out = item.repair()
            if inspect.isawaitable(out):
                await out
            if await _check(item):
                res.repaired.append(item.name)
                logger.info("機器を直した：%s", item.name)
            else:
                res.failed.append(item.name)
                logger.warning(
                    "機器を直せなかった：%s（%.0f 秒あとにまた試す）", item.name, retry_sec
                )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            res.failed.append(item.name)
            logger.warning("機器を確かめられなかった：%s", item.name, exc_info=True)
    return res


async def run(
    items: Sequence[Item],
    *,
    cfg: "DeviceWatchConfig | None" = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """見張りの仕事。起動の `first_delay_sec` あとから、間隔を決まりで変えながら確かめ続ける（止めるのは取り消し）。"""
    cfg = cfg or DeviceWatchConfig()
    await sleep(cfg.first_delay_sec)
    interval: "float | None" = None
    while True:
        res = await check_once(items, retry_sec=cfg.base_sec)
        interval = next_interval(interval, res, cfg=cfg)
        logger.debug("機器の見張り：次は %.0f 秒あと", interval)
        await sleep(interval)
