"""機器の見張りの決まり（知-bv・`設計方針_機器の見張り`）。純関数だけ。確かめと直し方は `io/device_watch.py`。"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import DeviceWatchConfig


@dataclass
class CheckResult:
    """見張り 1 回の結果。直した機器と、直せなかった機器の名前。どちらも空なら、確かめたものはすべて使えた。"""

    repaired: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def next_interval(prev: "float | None", result: CheckResult, *, cfg: DeviceWatchConfig) -> float:
    """次の確かめまでの秒。初めは `base_sec`。すべて使えていれば前の倍（最大 `max_sec`）、直したか直せなかったら
    `base_sec` に戻す。"""
    if prev is None or result.repaired or result.failed:
        return cfg.base_sec
    return min(prev * 2.0, cfg.max_sec)
