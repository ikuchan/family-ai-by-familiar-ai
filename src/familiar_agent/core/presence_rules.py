"""定点の静止物を人と数えない（知-v・2026-09-18・`知覚在席` v0.25）。

人検出（YOLO）は 1 枚の写真を独立に読み、人らしい形だけで数える。出入口の定点で右下の暗い塊
（家具か布）を 7 分間「1 人」と読み続け、在席表も `/speaker` の指定も切れなかった（実機 21:50〜
21:57・写真は同じ絵・動体イベントも無し）。閾値を上げるだけだと本当の人も落ちる。**人は動く**：
前回の枠と重なり（IoU）`min_iou` 以上で対応づく枠は「動かない時間」を引き継ぎ、`static_sec` 以上
動かない枠は数えない。動けば（対応づかなければ）0 から。動体イベントでも 0 から（`reset`）。
副作用：本当の人が `static_sec` まったく動かないと「居ない」になる。動けば戻る。

ここは純関数。状態（`StaticBoxes`）は定点ごとにセンサが持つ。
"""

from __future__ import annotations

from dataclasses import dataclass, field

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass(frozen=True)
class StaticBoxes:
    """前回見た枠と、それぞれが動かずに居る起点の時刻。"""

    boxes: tuple[tuple[Box, float], ...] = field(default_factory=tuple)


def iou(a: Box, b: Box) -> float:
    """2 つの枠の重なり（交わり／和）。"""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def reset(_state: StaticBoxes) -> StaticBoxes:
    """動体イベント：全部の積算を捨てる（次に見た枠はみな 0 から）。"""
    return StaticBoxes()


def count_moving(
    state: StaticBoxes, boxes: list[Box], *, now: float, static_sec: float, min_iou: float
) -> tuple[int, StaticBoxes]:
    """いま見えた枠のうち**人として数える数**と、次回に渡す状態。

    前回の枠と貪欲に対応づける（IoU の高い組から）。対応づいた枠は起点を引き継ぎ、
    `now − 起点 >= static_sec` なら数えない。対応づかなかった枠は起点＝`now`（数える）。
    """
    prev = list(state.boxes)
    pairs = sorted(
        ((iou(b, pb), i, j) for i, b in enumerate(boxes) for j, (pb, _) in enumerate(prev)),
        key=lambda t: -t[0],
    )
    since: dict[int, float] = {}
    used: set[int] = set()
    for score, i, j in pairs:
        if score < min_iou:
            break
        if i in since or j in used:
            continue
        since[i] = prev[j][1]
        used.add(j)
    nxt: list[tuple[Box, float]] = []
    counted = 0
    for i, b in enumerate(boxes):
        start = since.get(i, now)
        if now - start < static_sec:
            counted += 1
        nxt.append((b, start))
    return counted, StaticBoxes(boxes=tuple(nxt))
