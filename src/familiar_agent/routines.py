"""Routine and schedule helpers for quiet hours and continuation flow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(slots=True, frozen=True)
class QuietHoursRule:
    start_hour: int = 23
    end_hour: int = 7

    def is_quiet(self, now: datetime | None = None) -> bool:
        moment = now or datetime.now()
        if self.start_hour == self.end_hour:
            return False
        if self.start_hour < self.end_hour:
            return self.start_hour <= moment.hour < self.end_hour
        return moment.hour >= self.start_hour or moment.hour < self.end_hour


@dataclass(slots=True, frozen=True)
class RoutineDecision:
    quiet_hours: bool
    schedule_multiplier: float
    notes: tuple[str, ...] = ()


def quiet_hours_rule() -> QuietHoursRule:
    """静穏時間の規則を Config から作る。

    出所は**環境変数（`QUIET_HOURS_START`／`QUIET_HOURS_END`）→ Config の既定**の2段だけ。
    以前は `~/.familiar_ai/schedule.conf` と作業ディレクトリの `ROUTINES.md` も読んでいたが、
    どちらも存在せず、コード内の既定（23〜7）が効いているだけだった。出所が4段あると、
    どの値が効いているのかを確かめるのに4箇所を見ることになる。
    """
    from .config import AgentConfig

    cfg = AgentConfig()
    return QuietHoursRule(start_hour=cfg.quiet_hours_start, end_hour=cfg.quiet_hours_end)


def load_optional_notes(base_dir: Path | None = None) -> dict[str, str]:
    root = base_dir or Path.cwd()
    result: dict[str, str] = {}
    for name in ("SOUL.md", "TODO.md", "ROUTINES.md"):
        path = root / name
        if path.exists():
            result[name] = path.read_text(encoding="utf-8").strip()
    return result


def evaluate_routine_state(rule: QuietHoursRule, now: datetime | None = None) -> RoutineDecision:
    quiet = rule.is_quiet(now)
    return RoutineDecision(
        quiet_hours=quiet,
        schedule_multiplier=0.45 if quiet else 1.0,
        notes=("quiet-hours",) if quiet else (),
    )
