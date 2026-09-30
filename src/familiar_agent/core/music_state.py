"""音楽の状態（知-aa 段 1・2026-09-21）。

**溜めるのはここだけ**——鳴らし始めた時刻と、鳴っているという印と、最後に O に書いた曲名。
いまの曲名や音量は MPRIS から読む（`ユースケース④` [D-自己状態]）。印と時刻を持つのは、寿命
（30 分）と門（鳴っているあいだは音楽の話だけ通す）が、MPRIS を読まずに即答できる必要があるため。
最後に書いた曲名は、機器の状態の写しではなく、ループが O に書いた事実の控えである（曲が変わった
ときだけ書くため・知-aa 段 2・本人の決定イ）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MusicState:
    playing: bool = False
    started_at: float = 0.0
    last_title: str = ""
