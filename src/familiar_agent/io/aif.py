"""自律機構接続 AIF：T と I をつなぐ唯一の口（`設計図` ③-2）。

設計は口を4つ（IIF・DIF・AIF・OIF）と定め、この4つ以外にコンポーネントどうしが直接
つながる線を置かない。ところが AIF は実装として存在せず、**T と I が互いの中身に手を
伸ばしていた**。

- T→I の情動発火：`tonic.py` が `InformationProcessing.push_affect()` を直接呼ぶ
- I→T の Nudge：`agent.py` が `nudge_current_mood()` を直接呼ぶ

行き来をこの口へ集める。挙動は変えない。キューへの書き方も mood の計算もそのままで、
AIF は転送し、通ったものを debug ログに残す。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..mood_register import MoodPAD

logger = logging.getLogger(__name__)

# ログに載せる内声の長さ。記憶の内容と同じ扱いで、debug でも先頭だけにする。
_TRAIL_CHARS = 24


@dataclass(frozen=True)
class Firing:
    """T が起こした情動発火。I の入口へ渡る。

    `axis` は 5 欲求の軸名（`seeking`・`safety`・`bond`・`esteem`・`rest`）。
    `inner_voice` は Config が持つ軸ごとの文で、行動は指定しない（[D-行動選択]）。
    """

    axis: str
    inner_voice: str


@dataclass(frozen=True)
class Nudge:
    """I が返す作用。W の感情トーンで T の mood を動かす。

    `items` は (PAD, 重み) の並びで、想起した記憶と現ターンの感情が入る。重みは根づき
    （現ターンの感情は 1.0）。空でも通す（経過による減衰だけが効く）。
    """

    items: list = field(default_factory=list)


class AIF:
    """T と I の唯一の出入り口。

    `loop` は I（`InformationProcessing`）、`nudge` は T の mood レジスタを動かす関数
    （既定は `mood_register.nudge_current_mood`）。関数で受け取るのは、DB を触らずに
    試せるようにするためである。
    """

    def __init__(self, loop, *, nudge: Callable | None = None) -> None:
        self._loop = loop
        self._nudge = nudge

    def fire(self, firing: Firing) -> None:
        """T の発火を I へ渡す（T → I）。"""
        logger.debug("AIF fire ← %s／%s", firing.axis,
                     firing.inner_voice[:_TRAIL_CHARS])
        # I が受ける軸名は大文字。ここで揃えるのは、T 側が小文字の軸名で回している
        # ためで、変換を口の中に閉じておけば両側が相手の書き方を知らずに済む。
        self._loop.push_affect(firing.axis.upper(), firing.inner_voice)

    def nudge(self, nudge: Nudge) -> "MoodPAD":
        """I の作用を T の mood レジスタへ渡す（I → T）。新しい mood を返す。"""
        logger.debug("AIF nudge ← %d件", len(nudge.items))
        fn = self._nudge
        if fn is None:
            from ..mood_register import nudge_current_mood
            fn = nudge_current_mood
        got = fn(nudge.items)
        logger.debug("AIF nudge → (%.2f,%.2f,%.2f,%.2f)",
                     got.p, got.pn, got.a, got.dom)
        return got


__all__ = ["AIF", "Firing", "Nudge"]
