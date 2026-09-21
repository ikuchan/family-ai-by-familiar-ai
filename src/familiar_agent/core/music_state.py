"""音楽の状態（知-aa 段 1・2026-09-21）。

**溜めるのはここだけ**——鳴らし始めた時刻と、鳴っているという印。曲名や音量は MPRIS から
読む（`ユースケース④` [D-自己状態]）。この 2 つを持つのは、寿命（30 分）と門（鳴っている
あいだは音楽の話だけ通す）が、MPRIS を読まずに即答できる必要があるためである。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MusicState:
    playing: bool = False
    started_at: float = 0.0
