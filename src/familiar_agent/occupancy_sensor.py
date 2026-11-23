"""在/不在を測る器：撮って、どの定点かを決めて、人が居るかを数える。

`知覚在席` §3-2 の在/不在は **G（T 側・連続）** が担う。ここが部品を束ねる場所で、定点
（`poses`）、人検出（`recognition.person_detector`）、定点別の記録（`occupancy_map`）、
動体イベント（`recognition.motion_events`）を繋ぐ。**誰かは問わない**（#17 の担当）。

起こされ方は2つある。カメラが「動いた」と言ってきたときと、一定間隔（既定30秒）。動体
だけでは足りない。静止している人は動体を出さないので、動きが無いあいだも確かめる。

**どの定点を見ているか分からないときは、何も記録しない**（`知覚在席` §3-3 の振動中ゲート）。
向きが読めない、または定点から離れているときの映像は、どの定点のものでもない。撮れなかった
ときも同じで、「誰も居ない」にはしない。カメラの不調で人が消えることになる。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from typing import Any

from .poses import nearest_pose
from .core.aio import wait_within
from .core.occupancy_rules import StaticBoxes, count_moving, reset
from .occupancy_map import OccupancyMap
from . import view_change
from .visual_norm import cosine_distance, is_ready, update_ema

logger = logging.getLogger(__name__)

# 見比べ 1 回の締切（秒）。視覚のモデルが返らなくても、次の見比べを塞がない。
_VIEW_JUDGE_SEC = 30.0


#: 人を見なかった読みの種類（知-bg 段 1）。居なくなった行に、この順で回数を並べる。
_READINGS = (
    "枠なし",
    "動かないので数えない",
    "向きが読めない",
    "移動中",
    "撮れなかった",
    "確認に失敗",
)


def _delay_before(last_check: float, now: float, min_gap: float) -> float:
    """次の確認まで待つ秒数。前回から間が空いていれば 0（すぐ確かめる）。

    動いているあいだ動体イベントは毎秒何件も飛ぶ。そのたびに撮って YOLO を回すと、実機では
    **0.15 秒ごと**に確認していた。動き始めには即座に気づきたいので最初の1件は待たせず、
    続けて飛んでくるぶんだけ間引く。
    """
    return max(0.0, min_gap - (now - last_check))


class OccupancySensor:
    """定点ごとに人が居るかを見て、`OccupancyMap` を更新し続ける。"""

    def __init__(
        self,
        camera: Any,
        poses_getter: Callable[[], Any],
        detector: Any,
        *,
        tolerance: float,
        window_sec: float,
        interval_sec: float,
        min_gap_sec: float = 3.0,
        static_sec: float = 300.0,
        static_iou: float = 0.9,
        static_grace_sec: float = 0.0,
    ) -> None:
        self._camera = camera
        # 静止物を人と数えない（知-v）。定点ごとに前回の枠と起点を持つ。
        self._static_sec = static_sec
        self._static_iou = static_iou
        self._static_grace = static_grace_sec  # 枠が抜けても積算を持ち越す（知-v-ろ）
        self._static: dict[str, StaticBoxes] = {}
        self._last_reading: "tuple[str, int, float] | None" = (
            None  # 定点・人数・時刻（パネル用・環-p-ろ）
        )
        self._poses_getter = poses_getter
        self._detector = detector
        self._tolerance = tolerance
        self._window = window_sec
        self._interval = interval_sec
        self._min_gap = min_gap_sec
        self._last_check = float("-inf")
        self._map: OccupancyMap | None = None
        # 見えの「普通」。無ければ在席だけを見る（S4 を入れる前と同じ動き）。
        self._encoder: Any = None
        self._norm_store: Any = None
        # 前の見えと見比べ（知-i 段 2）。挿されなければ見比べない。
        self._view_store: Any = None
        self._view_judge: Any = None
        self._view_threshold = 0.0
        self._view_settled = 0.0
        self._view_cooldown_sec = 0.0
        #: 走っている見比べ（同時に 1 つだけ・見回りは待たない）。
        self._view_task: asyncio.Task | None = None
        self._scene_surprise: float | None = None
        self._frame_b64: str | None = None
        self._task: asyncio.Task | None = None
        # 動体イベントで即座に起こすための合図。間隔の待ちを飛び越える。
        self._wake = asyncio.Event()
        # 在席の切り替わりを 1 行残すための控え（知-bg 段 1）。最後に人を見た定点と時刻、そのあとの読みの種類ごとの回数。
        self._was_occupied = False
        self._last_person: "tuple[str, float] | None" = None
        self._emptied_at: "float | None" = None
        self._tally: dict[str, int] = dict.fromkeys(_READINGS, 0)

    # --- 外から読むもの ---------------------------------------------------

    def room_occupied(self) -> bool:
        return self._map.room_occupied(time.time()) if self._map else False

    def poses_seen(self) -> list[str]:
        return self._map.poses_seen(time.time()) if self._map else []

    def stalest_pose(self) -> str | None:
        """次に見に行くべき定点。見回り（S5）が使う。"""
        return self._map.stalest_pose(time.time()) if self._map else None

    def stale_order(self) -> "list[tuple[str, float | None]]":
        """定点を見ていない順に（`[いま]` の 1 行の材料・知-c）。地図が無ければ空。"""
        return self._map.stale_order(time.time()) if self._map else []

    def attach_visual_norm(self, encoder: Any, store: Any) -> None:
        """見えの「普通」を扱う部品を挿す（`知覚在席` §3-4 の見え層）。"""
        self._encoder = encoder
        self._norm_store = store

    def attach_view_change(
        self,
        store: Any,
        judge: Any,
        *,
        threshold: float,
        settled: float,
        cooldown_sec: float,
    ) -> None:
        """前の見えの置き場と、前と今の画像（JPEG のバイト列 2 つ）を見比べる関数を挿す（知-i 段 2）。

        `judge(before, after)` は `("物" | "明るさ" | "なし", 何が変わったか)` か None を返す。値は
        `config.camera.view_change_*` から渡す（既定はそこにだけ置く）。
        """
        self._view_store = store
        self._view_judge = judge
        self._view_threshold = threshold
        self._view_settled = settled
        self._view_cooldown_sec = cooldown_sec

    def scene_surprise(self) -> float | None:
        """直近の $\\widehat{S}_{景色}$（定点の「普通」からのコサイン距離）。

        「普通」が育つまで（既定5回）は `None`。比較対象が無いのに 0 を返すと、
        「いつもどおり」と区別が付かない。
        """
        return self._scene_surprise

    def latest_frame_b64(self) -> str | None:
        """直近に見たフレーム。GUI が在席確認のカメラ映像として表示する。"""
        return self._frame_b64

    def last_reading(self) -> "tuple[str, int, float] | None":
        """最新の読み（定点・人数・何秒前）。まだ見ていなければ None（パネル用・環-p-ろ）。"""
        if self._last_reading is None:
            return None
        pose, people, at = self._last_reading
        return (pose, people, max(0.0, time.time() - at))

    def last_person(self) -> "tuple[str, float] | None":
        """最後に人を見た定点と、それが何秒前か。まだ見ていなければ None（知-bg 段 2・名前で起こす判定のログ用）。"""
        if self._last_person is None:
            return None
        pose, at = self._last_person
        return (pose, max(0.0, time.time() - at))

    def on_motion(self) -> None:
        """カメラが「動いた」と言ってきた。次の間隔を待たずに確かめる。動いた証拠なので静止の積算は捨てる。"""
        self._static = {k: reset(v) for k, v in self._static.items()}
        self._wake.set()

    # --- 中身 -------------------------------------------------------------

    async def _ensure_map(self) -> OccupancyMap | None:
        """定点を毎回受け取り、地図の定点を合わせる（知-bw・2026-11-24）。

        定点は、カメラの ONVIF が繋がったときに読まれる（`EmbodiedAgent._on_camera_connected`）。起動のときにカメラが
        遅れると、あとから届く。受け取るのは覚えた一覧で、ここでカメラを待つことは無い。地図は作り直さず、定点の名前だけ
        を合わせる（見た時刻を持ち越す。作り直すと、居るのに「居なくなった」になる）。
        """
        poses = await self._poses_getter()
        if not poses:
            return None
        names = [p.name for p in poses]
        if self._map is None:
            self._map = OccupancyMap(names, window_sec=self._window)
        self._map.poses = names
        self._poses = poses
        return self._map

    async def check_once(self) -> str | None:
        """1回見る。記録した定点の名前を返す。記録しなかったときは `None`。"""
        name = await self._check()
        self._note_change(time.time())
        return name

    def _count(self, kind: str) -> None:
        self._tally[kind] = self._tally.get(kind, 0) + 1

    def _note_change(self, now: float) -> None:
        """在席が切り替わったときだけ INFO に 1 行残す（知-bg 段 1・2026-10-16・本人の決定ア）。

        名前で呼んだ声の 4 件が「居ない」になり、人が消えた理由（枠なし・動かない・撮れない・別の向き）が DEBUG にしか
        無かった。居なくなったらそのあいだの読みの回数を、また居るようになったら居なかった長さを残す。判定は変えない。
        """
        occupied = self.room_occupied()
        if self._was_occupied and not occupied:
            pose, seen_at = self._last_person or ("?", now)
            counts = "・".join(f"{k} {self._tally.get(k, 0)}" for k in _READINGS)
            logger.info(
                "在席：居なくなった（最後に人を見たのは %s・%.0f 秒前／そのあとの読み %s）",
                pose,
                now - seen_at,
                counts,
            )
            self._emptied_at = seen_at  # 居なかった長さは、最後に人を見た時刻から数える
        elif occupied and not self._was_occupied and self._emptied_at is not None:
            pose = self._last_person[0] if self._last_person else "?"
            logger.info(
                "在席：また居る（%s・居なかったのは %.0f 秒）", pose, now - self._emptied_at
            )
        self._was_occupied = occupied

    async def _check(self) -> str | None:
        try:
            pmap = await self._ensure_map()
            if pmap is None:
                return None
            position = await self._camera.position()
            if position is None:
                logger.debug("向きが読めないので在席を記録しない")
                self._count("向きが読めない")
                return None
            pose = nearest_pose(self._poses, position[0], position[1], self._tolerance)
            if pose is None:
                logger.debug("どの定点でもない向きなので在席を記録しない（移動中）")
                self._count("移動中")
                return None
            # 見回りの画像は 1 つのファイルに書き直す（時刻つきで残さない・環-bc）。
            frame_b64, path = await self._camera.capture(patrol=True)
            if not path:
                logger.warning("フレームを撮れなかったので在席を記録しない")
                self._count("撮れなかった")
                return None
            self._frame_b64 = frame_b64
            boxes = await self._detector.boxes(path)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("在席の確認に失敗した（記録しない）: %s", e)
            self._count("確認に失敗")
            return None
        seen = await self._update_scene_norm(pose.name, path)
        now = time.time()
        # 動かない枠は物（知-v）。数える前に落とす。
        people, self._static[pose.name] = count_moving(
            self._static.get(pose.name, StaticBoxes()),
            list(boxes),
            now=now,
            static_sec=self._static_sec,
            min_iou=self._static_iou,
            grace_sec=self._static_grace,
        )
        if self._scene_surprise is not None:
            # 知-i 段 1 の測り（2026-11-22〜24 に 41,331 回集めて段 2 の足切りを決めた）。毎回なので debug
            logger.debug(
                "見えの隔たり：%s は %.3f（枠 %d 個・動く人 %d 人）",
                pose.name,
                self._scene_surprise,
                len(boxes),
                people,
            )
        if len(boxes) > people:
            logger.debug(
                "在席：%s の枠 %d 個のうち %d 個は %.0f 秒以上動かないので数えない",
                pose.name,
                len(boxes),
                len(boxes) - people,
                self._static_sec,
            )
        if seen is not None:
            await self._watch_view(pose.name, seen, path, len(boxes), now)
        self._last_reading = (pose.name, int(people), now)
        if people > 0:
            pmap.mark_seen(pose.name, now)
            logger.info("在席：%s に %d 人", pose.name, people)
            self._last_person = (pose.name, now)
            self._tally = dict.fromkeys(_READINGS, 0)
        else:
            pmap.mark_checked(pose.name, now)
            logger.debug("在席：%s に誰も居ない", pose.name)
            self._count("動かないので数えない" if boxes else "枠なし")
        return pose.name

    async def _update_scene_norm(self, pose_name: str, path: str) -> list[float] | None:
        """見えの「普通」を1枚ぶん育て、そこからの隔たりを控える。

        見えが取れなくても在席の判定は続ける（人が居るかは YOLO が別に見ている）。
        1回抜けても EMA は次で追いつく。見えが取れなかった回は隔たりを無しにする（前の定点の値を残さない）。
        """
        self._scene_surprise = None
        if self._encoder is None or self._norm_store is None:
            return None
        try:
            seen = await self._encoder.embed(path)
            if seen is None:
                return None
            norm, observations = self._norm_store.load(pose_name)
            self._scene_surprise = (
                cosine_distance(norm, seen) if norm is not None and is_ready(observations) else None
            )
            self._norm_store.save(pose_name, update_ema(norm, seen), observations=observations + 1)
            return list(seen)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("見えの普通を更新できなかった（在席の判定は続ける）: %s", e)
            return None

    async def _watch_view(
        self, pose_name: str, seen: list[float], path: str, boxes: int, now: float
    ) -> None:
        """前の見えと比べ、変わった候補なら前と今の見比べを別の仕事で始める（知-i 段 2・`知覚在席` §4）。

        人が映っていない・落ち着いた見えだけを扱う（毎回 DB を読まないよう、先にここで落とす）。前の見えが
        まだ無ければ、この見えを前の見えにする。見比べは視覚のモデルを呼ぶので、見回りを待たせないよう
        1 回きりの仕事にし、同時には 1 つだけ走らせる。いまは記録するだけで、話題にはしない（段 2b）。
        """
        if self._view_store is None or self._view_judge is None:
            return
        surprise = self._scene_surprise
        if boxes > 0 or surprise is None or surprise > self._view_settled:
            return
        if self._view_task is not None and not self._view_task.done():
            return
        try:
            earlier = self._view_store.load(pose_name)
            if earlier is None:
                with open(path, "rb") as f:
                    self._view_store.save(pose_name, seen, f.read())
                logger.info("前の見えを残した：%s", pose_name)
                return
            if not view_change.is_candidate(
                ref=earlier.embedding,
                seen=seen,
                surprise=surprise,
                boxes=boxes,
                checked_at=earlier.checked_at,
                now=now,
                threshold=self._view_threshold,
                settled=self._view_settled,
                cooldown_sec=self._view_cooldown_sec,
            ):
                return
            with open(path, "rb") as f:
                image = f.read()  # 見回りの画像は次の回に書き直されるので、いま読んでおく
        except Exception:  # noqa: BLE001
            logger.exception("前の見えを読めなかった（在席の判定は続ける）")
            return
        self._view_task = asyncio.create_task(
            self._compare_view(pose_name, seen, earlier, image), name="view-compare"
        )

    async def _compare_view(
        self, pose_name: str, seen: list[float], earlier: Any, image: bytes
    ) -> None:
        """前と今を見比べる 1 回（締切 `_VIEW_JUDGE_SEC`）。読めた判定なら種類によらず前の見えを今に置き換え
        （明るさの違う前の見えと比べ続けない）、読めなければ見比べた時刻だけ残す（見回りのたびに呼び直さない）。"""
        distance = cosine_distance(earlier.embedding, seen)
        verdict = None
        try:
            verdict = await wait_within(self._view_judge(earlier.image, image), _VIEW_JUDGE_SEC)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.warning("前の見えと見比べられなかった：%s", pose_name, exc_info=True)
        try:
            if verdict is None:
                self._view_store.mark_checked(pose_name)
                logger.info(
                    "見比べた：%s は 読めなかった（前の見えから %.3f）", pose_name, distance
                )
                return
            self._view_store.save(pose_name, seen, image)
            if verdict[0] == "物":
                logger.info(
                    "見えが変わった：%s（前の見えから %.3f）%s", pose_name, distance, verdict[1]
                )
            else:
                logger.info(
                    "見比べた：%s は %s（前の見えから %.3f）", pose_name, verdict[0], distance
                )
        except Exception:  # noqa: BLE001
            logger.exception("見比べの結果を残せなかった：%s", pose_name)

    async def settle_view(self) -> None:
        """走っている見比べを待つ（試験と止めるとき）。"""
        if self._view_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._view_task

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="occupancy-sensor")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        if self._view_task is not None and not self._view_task.done():
            self._view_task.cancel()
            await self.settle_view()

    async def _run(self) -> None:
        while True:
            await self.check_once()
            self._last_check = time.monotonic()
            self._wake.clear()
            try:
                # 動体で起こされたら間隔を待たずに次を見る。
                # 止めるときのキャンセルを消さない待ち方（環-aa・3.11 の `wait_for` は消すことがある）。
                await wait_within(self._wake.wait(), self._interval)
            except (TimeoutError, asyncio.TimeoutError):
                continue  # 動きが無いまま間隔が来た
            delay = _delay_before(self._last_check, time.monotonic(), self._min_gap)
            if delay > 0:
                await asyncio.sleep(delay)
