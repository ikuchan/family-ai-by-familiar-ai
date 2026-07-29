"""T（自律機構 Tonic）の常駐タスク：時計を持つ唯一の側。

正本＝`設計図_Mermaid` ③・課題5 A節。役割分担は次のとおり。

- **T（ここ）**：時計を見る。$P_T$ ごとに drive を蓄積し、閾値で発火させ、**AIF 経由で QA へ積む**。
- **I（駆動体）**：時計を見ない。3キューを待ち、来たどれでも起きる。

drive の蓄積と発火判定は `core.drive_dynamics` の純関数が持ち、ここは時間で回して永続化し、
発火を QA へ渡すだけにする。同じ処理は GUI の描画ループにも埋まっていた（`gui.py:_tick_drives`）
一方で CUI には無く、自律が動く条件が入口ごとに違っていた。T を1本立てて揃える。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from datetime import datetime

from ..config import DriveConfig
from ..core import drive_dynamics as dd
from ..core.drive_autonomy import inner_voice_for, select_fired_axis
from ..drive_register import AiDrivers, load_drives, save_drives
from ..mood_register import load_current_mood
from .rest import run_rest_pass

logger = logging.getLogger(__name__)

# 課題5 A節の確定値。蓄積式 drive_i ← clip(drive_i + rate·mult·learn·g_D·P_T) にも入るので、
# 周期を変えると蓄積そのものが変わる。勝手に動かさない。
TONIC_PERIOD_SEC = 0.5

# 顔も声も照合できていない在席者の呼び名。居ることは分かるが誰かは分からない状態で、
# 「誰も居ない」とは区別する（用語一覧の二層：在/不在＝T、誰か＝I）。
UNIDENTIFIED = "誰か"


async def step_drives(dt: float) -> tuple[dd.DriveFiring, AiDrivers]:
    """1 tick 分の dynamics を回して永続化し、(発火, 蓄積後・放電前の drives) を返す。

    重い呼び出しではないが DB を触るのでスレッドへ逃がす。`load_current_mood` は内部で
    `db.lock` を取り再入できないため、ロックを取る前に読む（既存 GUI 実装と同じ順序）。
    """
    def _work() -> tuple[dd.DriveFiring, AiDrivers]:
        from ..db import get_db

        cfg = effective_drive_cfg(DriveConfig())    # 深夜は蓄積が遅くなる（#13）
        mood = load_current_mood()          # 自己接続でロックを取り、抜ける
        database = get_db()
        with database.lock:
            conn = database.conn()
            drives = load_drives(conn)
            accumulated = dd.accumulate(drives, mood, dt=dt, cfg=cfg)
            firing = dd.fired(accumulated, cfg)
            persisted = dd.discharge(accumulated, firing, cfg) if firing.any else accumulated
            save_drives(conn, persisted)
        return firing, accumulated

    return await asyncio.to_thread(_work)


def effective_drive_cfg(cfg: DriveConfig, now: datetime | None = None) -> DriveConfig:
    """いまの時刻に応じた設定を返す。静穏時間なら軸ごとの倍率へ差し替える（#13）。

    時計を見るのは T の役なので判定をここに置く（`core.drive_dynamics` は時計を持たない
    純関数として定義されている）。窓は静穏時間（`QUIET_HOURS_START`／`END`・既定 23〜7）を
    そのまま使う。「自分から話しかけない時間」と「欲求が募る速さ」は別の事柄だが、窓を
    二つ持つと、どちらが効いているかを二箇所で確かめることになる。

    **REST だけは抑えず、逆に募らせる。** 設計（`設計詳細：発火・mood 機構` §82）が
    「REST の募りは別途バイアス＋時間帯倍率（夜高い）」と定めるためで、一律に掛けると
    正反対になる。値の根拠は `DriveConfig.mult_quiet_rest` のコメントにある。
    """
    from ..routines import quiet_hours_rule

    if not quiet_hours_rule().is_quiet(now):
        return cfg
    return replace(cfg, mult=cfg.mult_quiet, mult_rest=cfg.mult_quiet_rest)


def _names(names: set[str]) -> str:
    """在席者の集合をログ用の1行にする。"""
    return "・".join(sorted(names)) or "（なし）"


class Tonic:
    """自律機構の常駐タスク。$P_T$ ごとに drive を進め、発火を QA へ積む。"""

    def __init__(self, information_processing, *, agent=None,
                 period: float = TONIC_PERIOD_SEC,
                 drive_cfg: DriveConfig | None = None,
                 presence=None) -> None:
        self._ip = information_processing
        self._agent = agent
        # 在/不在の情報源（`PresenceSensor`）。渡さなければ身元の情報源だけで判断する。
        # agent から取りに行くと、テストの MagicMock が「常に誰か居る」を返してしまう。
        self._presence = presence
        self._period = period
        # 前回の在席者。差分を取って人の出入りを QD へ積む。None＝まだ一度も見ていない
        # （起動直後に既に居る人を「たった今来た」と扱わないため、空集合と区別する）。
        self._present_names: set[str] | None = None
        # 自発の可否は `DRIVE5_AUTONOMOUS`（5欲求）で決める。旧 `DesireSystem`（15欲求）用の
        # `AUTO_DESIRE` とは系統が違うので独立させる。
        self._cfg = drive_cfg or DriveConfig()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    def scan_presence(self) -> None:
        """在席者の集合を見て、前回との差分を人の出入りとして QD へ積む。

        情報源は二層に分かれている（用語一覧）。**在/不在は `PresenceSensor`**（YOLO・登録が
        要らない）、**誰かは PMM**（顔の照合・`/speaker` の自己申告）。照合が済んでいなければ
        `UNIDENTIFIED` として扱い、居ることだけ伝える。

        読むのはメモリ上の値で、DB も I/O も触らない（センサは別の常駐タスクが更新する）。
        """
        agent = self._agent
        if agent is None:
            return
        try:
            rows = agent._pmm.presence_status()
        except Exception:  # noqa: BLE001
            rows = []
        current = {str(r.get("name") or r.get("person_id") or "") for r in rows}
        current.discard("")
        # 在/不在は YOLO（登録が要らない）、名前は照合が済んだときだけ。名前が分からない
        # ことと、誰も居ないことは別である。前者は「誰か」として、居ることだけ伝える。
        sensor = self._presence
        if sensor is not None:
            try:
                if sensor.room_occupied() and not current:
                    current = {UNIDENTIFIED}
            except Exception:  # noqa: BLE001
                logger.debug("在席センサを読めなかったので身元の情報源だけで判断する")
        previous_had_only_unidentified = self._present_names == {UNIDENTIFIED}
        if previous_had_only_unidentified and current and UNIDENTIFIED not in current:
            # 「誰か」で入室したあとに顔が照合できた。同じ人がそこに居続けているだけなので、
            # 退室は起きていない。素朴に差分を取ると、退室と入室が1件ずつ飛ぶ。
            logger.info("tonic 在席の身元が付いた：誰か → %s", _names(current))
            self._present_names = current
            return
        previous, self._present_names = self._present_names, current
        if previous is None:
            # 起動直後の1回目は差分を取らない。ただし「いま誰が見えているか」は残す。
            # これが無いと、イベントが出ないときに「T が回っていない」のか「誰も居ない」
            # のかを区別できない（在席イベントを確かめる手立てが無かった）。
            logger.debug("tonic 在席の初回走査：%s", "・".join(sorted(current)) or "誰も居ない")
            return
        # 保留していた発話を配るのは、在席がゼロから立ち上がった瞬間だけ。入室そのものは
        # 毎回積むが、会話中に家族が増えるたび保留が割り込むのは避ける。
        if current != previous:
            logger.info("tonic 在席の変化：%s → %s",
                        _names(previous), _names(current))
        rose_from_zero = not previous and bool(current)
        for name in sorted(current - previous):
            self._ip.push_device("入室", f"{name} が来た", release_pending=rose_from_zero)
            rose_from_zero = False    # 同時に2人来ても保留を配るのは1回
        for name in sorted(previous - current):
            self._ip.push_device("退室", f"{name} が居なくなった", release_pending=False)

    def _nobody_is_present(self) -> bool:
        """誰も居ないか。在/不在の層（`PresenceSensor`・YOLO・登録が要らない）で見る。

        身元の層（PMM の顔の照合）は使わない。顔が未登録なら、目の前に人が居ても
        「誰も居ない」になる（#15 で実機に出た欠陥）。

        **センサが無ければ偽を返す。** 「センサが無い」を「誰も居ない」と扱うと、カメラの
        無い構成で REST が常に内省へ落ち、人が居ても話しかけなくなる。
        """
        sensor = self._presence
        if sensor is None:
            return False
        try:
            return not sensor.room_occupied()
        except Exception:  # noqa: BLE001
            logger.debug("在席センサを読めなかったので、内省へは回さない")
            return False

    async def _run(self) -> None:
        last = time.monotonic()
        while True:
            try:
                await asyncio.sleep(self._period)
                now = time.monotonic()
                dt, last = now - last, now
                self.scan_presence()
                firing, accumulated = await step_drives(dt)
                if not firing.any:
                    continue
                if not self._cfg.autonomous:
                    logger.debug("Drive fired: %s（DRIVE5_AUTONOMOUS が off なので積まない）",
                                 select_fired_axis(firing, accumulated))
                    continue
                axis = select_fired_axis(firing, accumulated)
                if axis is None:
                    continue
                if axis == "rest" and self._nobody_is_present():
                    # 誰も居ないときの REST は内省へ回す（設計の折衷型・#2）。人へ話しかける
                    # 相手が居ないので、自発ターンにしても保留されるだけである。
                    logger.info("Drive fired: rest → 内省パスへ")
                    await run_rest_pass(self._agent)
                    continue
                prompt = inner_voice_for(axis, self._cfg)
                logger.info("Drive fired: %s → QA へ積む", axis)
                self._ip.push_affect(axis.upper(), prompt)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                # 自律は落とさない（アイドルが止まると何も起きなくなる）。
                logger.exception("tonic tick に失敗: %s", e)
