"""定点別 presence マップ：在/不在の層。

`知覚在席` §3-3 が定めるとおり、PTZ カメラは一度に1方向しか見ないので、**単一フレームの
不検出は部屋が空であることを意味しない**。在席は定点ごとの最終在席時刻として持ち、
部屋レベルの「誰か居る」は滞留窓での集約とする。空判定は、関連する定点を見て回って窓内に
誰も居ないときだけ成り立つ。

この層は**誰かを問わない**。YOLO が人を数えるだけで、顔の照合（#17）が無くても動く。
「誰か」は I 側の別の層が必要時に解く（用語一覧の二層）。

「人を見た」と「見たが居なかった」を別々に持つ。区別しないと、一度も向いていない定点が
不在と同じ扱いになり、部屋を空と判定してしまう。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PresenceMap:
    """定点ごとの「最後に人を見た時刻」と「最後に見に行った時刻」。

    `window_sec` は滞留窓（`課題5` §I の在席 timeout＝180 秒・2026-09-17 に 120 から。カメラは 1 箇所しか見ないので長めに）。人が静止していると動体も
    出ず毎フレームは検出されないので、窓の内側は居るものとして扱う。
    """

    poses: list[str]
    window_sec: float = 180.0
    _seen: dict[str, float] = field(default_factory=dict)
    _checked: dict[str, float] = field(default_factory=dict)

    def mark_seen(self, pose: str, at: float) -> None:
        """その定点で人を見た。見た以上、そこを見に行ったことでもある。"""
        if pose not in self.poses:
            logger.warning("定点の一覧に無い名前なので在席を捨てる: %.20s", pose)
            return
        self._seen[pose] = at
        self._checked[pose] = at

    def mark_checked(self, pose: str, at: float) -> None:
        """その定点を見たが、人は居なかった。"""
        if pose not in self.poses:
            logger.warning("定点の一覧に無い名前なので確認を捨てる: %.20s", pose)
            return
        self._checked[pose] = at

    def poses_seen(self, now: float) -> list[str]:
        """窓の内側で人を見た定点。"""
        return [p for p in self.poses if now - self._seen.get(p, float("-inf")) <= self.window_sec]

    def room_occupied(self, now: float) -> bool:
        """部屋に誰か居るか。いずれかの定点が窓の内側なら居る。"""
        return bool(self.poses_seen(now))

    def stale_order(self, now: float) -> "list[tuple[str, float | None]]":
        """見ていない順に (定点, 最後に見てからの秒・一度も無ければ None)。見回りの材料。"""
        return stale_order(self, now)

    def stalest_pose(self, now: float) -> str | None:
        """次に見に行くべき定点＝最も長く見ていないもの。

        見回り（S5）が巡回先を選ぶのに使う。一度も見ていない定点が最優先になる。
        """
        if not self.poses:
            return None
        return min(self.poses, key=lambda p: self._checked.get(p, float("-inf")))


def stale_order(pmap: PresenceMap, now: float) -> "list[tuple[str, float | None]]":
    """定点を**見ていない順**に並べる（一度も見ていないものが先頭・次に古い順）。

    `[いま]` の「見ていない順」の 1 行と、`look` の行き先の既定（`stalest_pose`）の材料。
    見た印は W に浮かないことがある（印が無い定点は想起に掛からない・2026-08-01）ので、
    機械が持つ時刻から出す。
    """

    def key(p: str) -> tuple[int, float]:
        at = pmap._checked.get(p)
        return (0, 0.0) if at is None else (1, -(now - at))

    out: list[tuple[str, float | None]] = []
    for p in sorted(pmap.poses, key=key):
        at = pmap._checked.get(p)
        out.append((p, None if at is None else max(0.0, now - at)))
    return out
