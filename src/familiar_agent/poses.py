"""定点：見回り・在席マップ・norm が共有する N 個の向き。

`知覚在席` §3-3 が定めるとおり、PTZ カメラは一度に1方向しか見ないので、**単一フレームの
不検出は部屋が空であることを意味しない**。在席も見えの「普通」も定点ごとに持ち、現在の
向きを最寄りの定点へ対応づける。どの定点からも離れていれば移動中とみなし、在席と norm を
更新しない（振動中ゲート・[D-向き]）。

定点は**カメラのプリセットと Config の和**である。プリセットは人がカメラのアプリで足せる
ので、後から増えたものが自動で定点になる。しきい値内に重なる向きは1つに畳み、Config 側の
名前を残す。

値は ONVIF の正規化座標 $[-1, 1]$ で、角度ではない。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ONVIF の絶対 pan/tilt 空間（実機で確認：pan・tilt とも [-1, 1]）。
_LIMIT = 1.0


@dataclass(frozen=True)
class Pose:
    """一つの定点。`name` は人が読む名前で、想起や発話にそのまま出る。"""

    name: str
    pan: float
    tilt: float


def parse_poses(text: str) -> list[Pose]:
    """`名前:pan,tilt;名前:pan,tilt` を読む。

    読めない項目は落として次へ進む。設定の誤記でカメラごと止めるより、読めた定点だけで
    動くほうがよい（定点ゼロでも見ること自体はできる）。
    """
    poses: list[Pose] = []
    for chunk in text.split(";"):
        item = chunk.strip()
        if not item:
            continue
        try:
            name, values = item.split(":", 1)
            pan_s, tilt_s = values.split(",", 1)
            pan, tilt = float(pan_s), float(tilt_s)
        except ValueError:
            logger.warning("定点の設定を読めなかったので飛ばす: %.40s", item)
            continue
        if abs(pan) > _LIMIT or abs(tilt) > _LIMIT:
            logger.warning("定点が可動範囲の外なので飛ばす: %.40s", item)
            continue
        poses.append(Pose(name.strip(), pan, tilt))
    return poses


def _distance(a: Pose, b: Pose) -> float:
    return math.hypot(a.pan - b.pan, a.tilt - b.tilt)


def merge_poses(config: list[Pose], presets: list[Pose], tolerance: float) -> list[Pose]:
    """Config の定点とカメラのプリセットを合わせる。Config が主で、プリセットが補い。

    しきい値内に重なるものは同じ定点として畳む。畳まないと、同じ向きに「普通」と在席マップが
    2つずつでき、どちらも育たない。名前は Config 側を残す（プリセットを足したときに Config を
    書き換える手作業を要らなくする）。
    """
    merged: list[Pose] = list(config)
    for extra in presets:
        near = next((p for p in merged if _distance(p, extra) <= tolerance), None)
        if near is not None:
            logger.debug("定点が重なるので畳む: %s ← %s", near.name, extra.name)
            continue
        merged.append(extra)
    return merged


async def build_pose_registry(config_text: str, camera, tolerance: float) -> list[Pose]:
    """定点一覧を組む。Config を読み、カメラのプリセットを足す。

    在席マップ・norm・見回りが同じ一覧を使う必要があるので、一箇所で組んで配る。
    プリセットは起動のたびに読み、人が後から足したものを次の起動から定点にする。
    カメラが無い、あるいはプリセットを読めないときは Config の分だけで動く。
    """
    poses = parse_poses(config_text)
    presets: list[Pose] = []
    if camera is not None:
        try:
            presets = await camera.presets()
        except Exception as e:  # noqa: BLE001
            logger.warning("カメラのプリセットを読めなかったので設定の定点だけ使う: %s", e)
    merged = merge_poses(poses, presets, tolerance)
    logger.info("定点 %d 件（設定 %d ＋ プリセット %d）：%s",
                len(merged), len(poses), len(presets),
                "、".join(p.name for p in merged) or "なし")
    return merged


def nearest_pose(poses: list[Pose], pan: float, tilt: float,
                 tolerance: float) -> Pose | None:
    """いまの向きに対応する定点。どこからも離れていれば `None`（移動中）。

    絶対移動には誤差があるので、少しのずれは同じ定点として吸収する。厳密に一致を求めると、
    到着するたびに移動中と判定されて「普通」が育たない。
    """
    here = Pose("", pan, tilt)
    best = min(poses, key=lambda p: _distance(p, here), default=None)
    if best is None or _distance(best, here) > tolerance:
        return None
    return best
