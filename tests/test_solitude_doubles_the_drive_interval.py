"""人と会話しないかぎり、欲求の発火間隔は倍々に伸びる（情-d）。

SEEKING・SAFETY・BOND の 3 軸に「ひとりの回数」n_i を持ち、蓄積の倍率に 2^(−n_i) を掛ける。
発火で n_i += 1、人の発話で全部 0 へ戻る。REST・ESTEEM は持たない。実機で SEEKING が
1 時間に 16 回起き、費用の大半になっていた。
"""

from __future__ import annotations

import asyncio
import os

import psycopg2
import pytest

from familiar_agent.config import DriveConfig
from familiar_agent.core import drive_dynamics as dd
from familiar_agent.core.solitude import Solitude, next_interval_minutes
from familiar_agent.drive_register import AiDrivers, load_solitude, save_drives, save_solitude
from familiar_agent.mood_register import MoodPAD

_NEUTRAL = MoodPAD()


def test_the_rate_halves_for_each_lonely_firing() -> None:
    cfg = DriveConfig()
    plain = dd.accumulate(AiDrivers(), _NEUTRAL, dt=60.0, cfg=cfg)
    lonely = dd.accumulate(AiDrivers(), _NEUTRAL, dt=60.0, cfg=cfg, solitude=Solitude(seeking=3))
    assert lonely.seeking == pytest.approx(plain.seeking / 8)
    assert lonely.safety == pytest.approx(plain.safety)


def test_rest_and_esteem_are_untouched() -> None:
    cfg = DriveConfig()
    plain = dd.accumulate(AiDrivers(), _NEUTRAL, dt=60.0, cfg=cfg)
    lonely = dd.accumulate(
        AiDrivers(), _NEUTRAL, dt=60.0, cfg=cfg, solitude=Solitude(seeking=5, safety=5, bond=5)
    )
    assert lonely.rest == pytest.approx(plain.rest)
    assert lonely.esteem == pytest.approx(plain.esteem)


def test_a_firing_counts_and_a_conversation_resets() -> None:
    s = Solitude()
    s = s.fired("seeking").fired("seeking").fired("safety")
    assert (s.seeking, s.safety, s.bond) == (2, 1, 0)
    s = s.fired("rest")  # 数えない
    assert s.seeking == 2
    s = s.reset(at=123.0)
    assert (s.seeking, s.safety, s.bond, s.reset_at) == (0, 0, 0, 123.0)


def test_the_count_is_capped() -> None:
    s = Solitude(seeking=20).fired("seeking")
    assert s.seeking == 20


def test_next_interval_is_the_base_times_two_to_the_n() -> None:
    cfg = DriveConfig()
    assert next_interval_minutes("seeking", Solitude(), cfg) == pytest.approx(5.0, rel=0.05)
    assert next_interval_minutes("seeking", Solitude(seeking=3), cfg) == pytest.approx(
        40.0, rel=0.05
    )


def test_solitude_round_trips_through_agent_state() -> None:
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.autocommit = True
    save_solitude(conn, Solitude(seeking=2, safety=1, bond=0, reset_at=99.5))
    got = load_solitude(conn)
    conn.close()
    assert got == Solitude(seeking=2, safety=1, bond=0, reset_at=99.5)


def test_the_tick_counts_a_firing_and_a_human_voice_resets(monkeypatch) -> None:
    """T の tick：発火で n が増え、人の発話（`_last_human_at`）が新しければ 0 へ戻る。"""
    from familiar_agent.loop import tonic as tonic_module

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.autocommit = True
    # 発火寸前の SEEKING と、ひとり 1 回目の状態から始める。
    save_drives(conn, AiDrivers(seeking=0.9995))
    save_solitude(conn, Solitude(seeking=1, reset_at=1000.0))
    monkeypatch.setattr(tonic_module, "effective_drive_cfg", lambda cfg, now=None: cfg)

    firing, _ = asyncio.run(tonic_module.step_drives(60.0, last_human_at=None))
    assert firing.seeking
    assert load_solitude(conn).seeking == 2

    # 人が話しかけた（reset_at より新しい）→ 0 へ戻り、reset_at がその時刻になる。
    asyncio.run(tonic_module.step_drives(0.5, last_human_at=2000.0))
    got = load_solitude(conn)
    conn.close()
    assert (got.seeking, got.reset_at) == (0, 2000.0)


def test_the_silence_default_is_an_hour() -> None:
    from familiar_agent.config import AgentConfig

    assert AgentConfig().silence_minutes == 60
