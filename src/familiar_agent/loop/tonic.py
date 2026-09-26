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
import contextlib
import logging
import time
from dataclasses import replace
from datetime import datetime

from ..config import DriveConfig
from ..core import drive_dynamics as dd
from ..io.dif import DIF
from ..io.aif import AIF, Firing
from ..core.drive_autonomy import inner_voice_for, select_fired_axis
from ..core.solitude import AXES, next_interval_minutes
from ..drive_register import AiDrivers, load_drives, load_solitude, save_drives, save_solitude
from ..mood_register import load_current_mood
from . import alarm_watch, music_watch, notes_watch, stopwatch_watch, timer_watch
from .rest import run_rest_pass

logger = logging.getLogger(__name__)

# 課題5 A節の確定値。蓄積式 drive_i ← clip(drive_i + rate·mult·learn·g_D·P_T) にも入るので、
# 周期を変えると蓄積そのものが変わる。勝手に動かさない。
TONIC_PERIOD_SEC = 0.5

# 顔も声も照合できていない在席者の呼び名。居ることは分かるが誰かは分からない状態で、
# 「誰も居ない」とは区別する（用語一覧の二層：在/不在＝T、誰か＝I）。
UNIDENTIFIED = "誰か"


async def step_drives(
    dt: float, *, last_human_at: float | None = None
) -> tuple[dd.DriveFiring, AiDrivers]:
    """1 tick 分の dynamics を回して永続化し、(発火, 蓄積後・放電前の drives) を返す。

    重い呼び出しではないが DB を触るのでスレッドへ逃がす。`load_current_mood` は内部で
    `db.lock` を取り再入できないため、ロックを取る前に読む（既存 GUI 実装と同じ順序）。

    `last_human_at`（人が最後に話しかけた epoch 秒・情-d）：ひとりの回数の `reset_at` より
    新しければ全軸 0 へ戻す（会話で基準の間隔へ戻る）。発火した軸は数える。
    """

    def _work() -> tuple[dd.DriveFiring, AiDrivers]:
        from ..db import get_db

        cfg = effective_drive_cfg(DriveConfig())  # 深夜は蓄積が遅くなる（#13）
        mood = load_current_mood()  # 自己接続でロックを取り、抜ける
        database = get_db()
        with database.lock:
            conn = database.conn()
            drives = load_drives(conn)
            lonely = load_solitude(conn)
            if last_human_at is not None and (
                lonely.reset_at is None or last_human_at > lonely.reset_at
            ):
                lonely = lonely.reset(at=last_human_at)
            accumulated = dd.accumulate(drives, mood, dt=dt, cfg=cfg, solitude=lonely)
            firing = dd.fired(accumulated, cfg)
            persisted = dd.discharge(accumulated, firing, cfg) if firing.any else accumulated
            save_drives(conn, persisted)
            for axis in AXES:
                if getattr(firing, axis):
                    lonely = lonely.fired(axis)
            save_solitude(conn, lonely)
            # **ここで確定する。** 共有接続は autocommit ではなく、`save_drives` も
            # `save_solitude` も commit しない。以前は次に誰かが commit するまで drive5 が
            # 宙に浮いていた（別の接続からは見えない・落ちれば消える）。
            conn.commit()
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

    def __init__(
        self,
        information_processing,
        *,
        agent=None,
        period: float = TONIC_PERIOD_SEC,
        drive_cfg: DriveConfig | None = None,
        presence=None,
    ) -> None:
        self._ip = information_processing
        # T は I の中身を直接呼ばない。行き来は AIF（自律機構接続）へ集める
        # （`設計図` ③-2 の4つの口）。
        self._aif = AIF(information_processing)
        # 人の出入りはカメラが出す機器の出来事なので、QD＝DIF を通す（環-e-は）。
        self._dif = DIF(ip=information_processing)
        self._agent = agent
        # 在/不在の情報源（`PresenceSensor`）。渡さなければ身元の情報源だけで判断する。
        # agent から取りに行くと、テストの MagicMock が「常に誰か居る」を返してしまう。
        self._presence = presence
        self._period = period
        # ストップウォッチの寿命を見た時刻と、その背景タスク（知-u）。
        self._stopwatch_checked = float("-inf")
        #: 音楽の寿命をいつ見たか（知-aa・30 秒ごと）。
        self._music_checked = float("-inf")
        self._background: set = set()
        # 前回の在席者。差分を取って人の出入りを QD へ積む。None＝まだ一度も見ていない
        # （起動直後に既に居る人を「たった今来た」と扱わないため、空集合と区別する）。
        self._present_names: set[str] | None = None
        self._unoccupied_since: float | None = (
            None  # センサが「誰も居ない」を見始めた時刻（在席表の失効）
        )
        # 自発の可否は `DRIVE5_AUTONOMOUS`（5欲求）で決める（旧 15 欲求の系は環-d で撤去）。
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
        self._expire_presence_table()
        self._expire_speaker()
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
        # **人の出入りでは話しかけない**（出-as §2.7・2026-09-26）。求めは立てず、記憶に記録だけする。
        # 居なくなったことは情動の側（bond・esteem が減る）で受ける。
        if current != previous:
            logger.info("tonic 在席の変化：%s → %s", _names(previous), _names(current))
        for name in sorted(current - previous):
            self._dif.record("入室", f"{name} が来た")
        for name in sorted(previous - current):
            self._dif.record("退室", f"{name} が居なくなった")

    def _expire_presence_table(self) -> None:
        """在席表（PMM・`/speaker`・顔照合）の失効。

        在席表には入る口だけあって出る口が無く、`/speaker パパ` が永久に残った（2026-09-17
        実機・カメラが 2 分「誰も居ない」でも自発が出た）。センサが「誰も居ない」を
        `presence_expire_sec` 見続けたら在席表を空にする（`mark_absent`）。話者の指定は別の寿命（知-t）。
        センサが無い構成では失効しない（在席表が唯一の情報源）。
        """
        sensor = self._presence
        agent = self._agent
        if sensor is None or agent is None:
            return
        try:
            occupied = bool(sensor.room_occupied())
        except Exception:  # noqa: BLE001
            return
        now = time.time()
        if occupied:
            self._unoccupied_since = None
            return
        if self._unoccupied_since is None:
            self._unoccupied_since = now
            return
        raw = getattr(getattr(agent, "config", None), "presence_expire_sec", None)
        expire = float(raw) if isinstance(raw, (int, float)) and raw > 0 else 60.0
        # `/speaker` を打ったら、そこから数え直す（知-s・2026-09-18 12:32 実機：打った 0.5 秒後に
        # 「誰も居ないが 340 秒」で失効した）。打った人はそこに居る。
        since = self._unoccupied_since
        set_at = getattr(agent, "_speaker_set_at", None)
        if isinstance(set_at, (int, float)) and set_at > since:
            since = float(set_at)
        if now - since < expire:
            return
        try:
            # **名前の分からない在席者も外す**（出-am・2026-09-22）。`get_present_ids()` は
            # 観測の `participants` になる口なので札を含まない（出-ae-は）。失効はここで
            # 在席表を空にするのが仕事なので、札まで含めた鍵を使う。
            ids = list(agent._pmm.present_keys())
        except Exception:  # noqa: BLE001
            return
        if not ids:
            return
        for pid in ids:
            agent._pmm.mark_absent(pid)
        logger.info(
            "tonic 在席表を失効：%d 人（誰も居ないが %.0f 秒）",
            len(ids),
            now - since,
        )

    def _expire_speaker(self) -> None:
        """話者の指定の寿命（知-t・2026-09-18）。分からなくなっていたら「不明」に戻す。

        判定は `agent.speaker_known()` の 1 箇所（`/speaker` から 60 秒・返事から 60 秒・顔照合）。
        戻すのは名前（`PersonRegistry`）と id（`PersonMemoryManager._speaker_id`）の両方。
        """
        agent = self._agent
        if agent is None:
            return
        try:
            sid = agent._pmm.current_speaker_id
            if not sid or agent.speaker_known():
                return
            name = getattr(agent._persons, "active_name", "")
            agent._persons.reset_to_default()
            agent._pmm.clear_speaker()
            logger.info("tonic 話者の指定が切れた：%s（60 秒返事が無く、顔も見ていない）", name)
        except Exception:  # noqa: BLE001
            logger.debug("話者の指定の寿命を見られなかった")

    def _solitude_note(self, axis: str) -> str:
        """ログ用：ひとり何回目で、次はおよそ何分後か（情-d）。読めなければ空。"""
        if axis not in AXES:
            return ""
        try:
            from ..db import get_db

            database = get_db()
            with database.lock:
                lonely = load_solitude(database.conn())
        except Exception:  # noqa: BLE001
            return ""
        cfg = effective_drive_cfg(self._cfg)
        return (
            f"（ひとり {getattr(lonely, axis)} 回目・次は約"
            f" {next_interval_minutes(axis, lonely, cfg):.0f} 分後）"
        )

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

    def _notes_due(self, *, now: float) -> bool:
        """パジュ宛てのメモを読む頃合いか（`notes_watch.INTERVAL_SEC` に 1 回・知-g-ろ）。"""
        last = getattr(self, "_notes_checked_at", None)
        return (
            last is None or now - last >= notes_watch.INTERVAL_SEC
        )  # 起動直後に 1 回（前回値を持つため）

    async def _maybe_check_notes(self, now: float) -> None:
        if not self._notes_due(now=now):
            return
        self._notes_checked_at = now
        try:
            await notes_watch.check_notes(self._ip)
        except Exception as e:  # noqa: BLE001
            logger.warning("パジュへのメモの確認に失敗: %s", e)

    def _fire_timers(self) -> None:
        """due を過ぎたタイマーを鳴らす（知-n）。器が無ければ何もしない。"""
        tool = getattr(self._agent, "_timer_tool", None)
        if tool is None:
            return
        try:
            cfg = getattr(self._agent, "config", None)
            ring = getattr(cfg, "timer_ring_sec", 0.0)
            gain = getattr(cfg, "timer_voice_gain", 1.0)
            quiet = False
            with contextlib.suppress(Exception):
                quiet = bool(self._agent._in_quiet_hours())
            with contextlib.suppress(Exception):  # 山谷の形（8 秒で小さく・25 秒で戻す）
                self._ip._dif.configure_ring(
                    soft_after=getattr(cfg, "ring_soft_after_sec", 8.0),
                    soft_until=getattr(cfg, "ring_soft_until_sec", 25.0),
                    soft_gain=getattr(cfg, "ring_soft_gain", 0.1),
                )
            timer_watch.fire_due(
                tool.store(),
                self._ip._dif,
                ring_sec=float(ring) if isinstance(ring, (int, float)) else 0.0,
                quiet=quiet,
                gain=float(gain) if isinstance(gain, (int, float)) else 1.0,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("タイマーの確認に失敗: %s", e)
        # アラーム（知-q・別物・別の器）。同じ tick から。
        alarm_tool = getattr(self._agent, "_alarm_tool", None)
        if alarm_tool is not None:
            try:
                cfg = getattr(self._agent, "config", None)
                ring = getattr(cfg, "alarm_ring_sec", 0.0)
                gain = getattr(cfg, "timer_voice_gain", 1.0)
                quiet = False
                with contextlib.suppress(Exception):
                    quiet = bool(self._agent._in_quiet_hours())
                alarm_watch.fire_due(
                    alarm_tool.store(),
                    self._ip._dif,
                    ring_sec=float(ring) if isinstance(ring, (int, float)) else 0.0,
                    quiet=quiet,
                    gain=float(gain) if isinstance(gain, (int, float)) else 1.0,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("アラームの確認に失敗: %s", e)
        # ストップウォッチの寿命（知-u・別物・別の器）。同じ tick からだが、表を見るのは 60 秒に 1 度で足りる。
        sw_tool = getattr(self._agent, "_stopwatch_tool", None)
        if sw_tool is not None and time.monotonic() - self._stopwatch_checked >= 60.0:
            self._stopwatch_checked = time.monotonic()
            cfg = getattr(self._agent, "config", None)
            max_sec = getattr(cfg, "stopwatch_max_sec", 6 * 3600.0)
            task = asyncio.ensure_future(
                stopwatch_watch.expire(
                    sw_tool.store(),
                    self._agent._oif,
                    max_sec=float(max_sec) if isinstance(max_sec, (int, float)) else 6 * 3600.0,
                )
            )
            self._background.add(task)
            task.add_done_callback(self._background.discard)
        # 音楽の寿命（知-aa・30 分）。同じ tick から。止めたら一言言う（本人の決定）。
        music_tool = getattr(self._agent, "_music_tool", None)
        if music_tool is not None and time.monotonic() - self._music_checked >= 30.0:
            self._music_checked = time.monotonic()
            music_task = asyncio.ensure_future(self._check_music())
            self._background.add(music_task)
            music_task.add_done_callback(self._background.discard)
        # 沈黙が期限切れで明けたのに何も届かないとき、まとめの求めを起こす（情-h）。時計は T。
        with contextlib.suppress(Exception):
            self._ip.check_silence_lifted()

    async def _check_music(self) -> None:
        """30 分たっていたら止めて一言言う（知-aa）。鳴っていなければ何もしない。"""
        tool = getattr(self._agent, "_music_tool", None)
        state = getattr(self._agent, "_music_state", None)
        if tool is None or state is None:
            return
        with contextlib.suppress(Exception):
            await music_watch.check_music_expired(
                io=tool._io,
                bus=tool._bus(),
                state=state,
                now=time.time(),
                say=lambda text: self._dif.device("音楽", text, passes_gate=True),
            )

    async def _run(self) -> None:
        last = time.monotonic()
        while True:
            try:
                await asyncio.sleep(self._period)
                now = time.monotonic()
                dt, last = now - last, now
                self.scan_presence()
                await self._maybe_check_notes(now)
                self._fire_timers()
                firing, accumulated = await step_drives(
                    dt, last_human_at=getattr(self._agent, "_last_human_at", None)
                )
                if not firing.any:
                    continue
                if not self._cfg.autonomous:
                    logger.debug(
                        "Drive fired: %s（DRIVE5_AUTONOMOUS が off なので積まない）",
                        select_fired_axis(firing, accumulated),
                    )
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
                logger.info("Drive fired: %s → QA へ積む%s", axis, self._solitude_note(axis))
                self._aif.fire(Firing(axis=axis, inner_voice=prompt))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                # 自律は落とさない（アイドルが止まると何も起きなくなる）。
                logger.exception("tonic tick に失敗: %s", e)
