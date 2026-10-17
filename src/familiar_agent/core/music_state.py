"""音楽の状態（知-aa 段 1・2026-09-21）。

**溜めるのはここだけ**——鳴らし始めた時刻と、鳴っているという印と、最後に O に書いた曲名。
いまの曲名や音量は MPRIS から読む（`ユースケース④` [D-自己状態]）。印と時刻を持つのは、寿命
（30 分）と門（鳴っているあいだは音楽の話だけ通す）が、MPRIS を読まずに即答できる必要があるため。
最後に書いた曲名は、機器の状態の写しではなく、ループが O に書いた事実の控えである（曲が変わった
ときだけ書くため・知-aa 段 2・本人の決定イ）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MusicState:
    playing: bool = False
    started_at: float = 0.0
    last_title: str = ""
    # 減音の戻しの予約（出-bd ①）。動いているあいだだけの控えで、DB には書かない。予約中の次の声は
    # 予約を取り消し、絞ったまま最初に読んだ基準（`ducked_base`）を引き継ぐ。
    restore_task: Any = None
    ducked_base: "float | None" = None


# ── ふだんの音量（知-bi・2026-10-17・本人の決定イ）────────────────────────────
# 人が声で変えたときと、止めたときの音量を DB に 1 つ持ち、かけ始めに使う。いまの音量は MPRIS から読むが、
# spotifyd は起動し直すと初期値（50%）に戻るので、人が選んだ値はここに残す。

VOLUME_KEY = "music_volume"


def saved_volume() -> "float | None":
    """記録したふだんの音量（0〜1）。無い・読めなければ None。"""
    from . import state_json

    v = state_json.read(VOLUME_KEY)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return max(0.0, min(1.0, float(v)))


def save_volume(v: float) -> bool:
    from . import state_json

    return state_json.write(VOLUME_KEY, round(max(0.0, min(1.0, float(v))), 3))


def base_volume(state: Any, status: "dict | None") -> "float | None":
    """絞る前の音量。絞っている最中は `ducked_base`、そうでなければ読んだ音量。読めなければ None。

    声で頼むと合図の声で先に絞られるので、読んだ音量をそのまま使うと絞った値を記録してしまう。
    """
    base = getattr(state, "ducked_base", None)
    if isinstance(base, (int, float)):
        return float(base)
    if status and isinstance(status.get("volume"), (int, float)):
        return float(status["volume"])
    return None
