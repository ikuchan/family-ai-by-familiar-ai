"""Drive dynamics（蓄積・気分変調・発火・放電）＝純関数（Slice 1・発火mood §2）。"""

from __future__ import annotations

import pytest
from dataclasses import replace

from familiar_agent.config import DriveConfig
from familiar_agent.core.drive_dynamics import (
    accumulate,
    discharge,
    fired,
    g_d,
    tick,
)
from familiar_agent.drive_register import AiDrivers
from familiar_agent.mood_register import REST_PAD, MoodPAD

_CFG = DriveConfig()


# ── g_D(M)：中立 mood で g_{D,i}=b_i（式の確定点） ─────────────────────────────

def test_g_d_at_rest_equals_bias():
    # 平静 → Σ C·(logit(x)−logit(rest))=0 → g=b_i。平静は軸ごと（案A）。
    g = g_d(MoodPAD())
    assert g.seeking == pytest.approx(_CFG.bias_seeking, abs=1e-9)
    assert g.safety == pytest.approx(_CFG.bias_safety, abs=1e-9)
    assert g.bond == pytest.approx(_CFG.bias_bond, abs=1e-9)
    assert g.esteem == pytest.approx(_CFG.bias_esteem, abs=1e-9)
    assert g.rest == pytest.approx(_CFG.bias_rest, abs=1e-9)


# ── g_D(M)：気分の向きに応じて募る（設計の性格） ─────────────────────────────

def test_g_d_direction_matches_design():
    """気分の向きと欲求の募り方の対応。**平静との比較でなく、両端の比較で見る。**

    案A で P と Pn の平静が 0.10 へ動き、この2軸は平静より下の幅（0〜0.10）が上の幅
    （0.10〜1.0）より狭くなった。「平静より上か下か」で書くと、軸ごとに使える幅が
    違うぶん閾値の当て方が難しくなる。両端どうしを比べれば、平静をどこに置いても
    向きだけを問える。
    """
    r = REST_PAD

    def m(**over):
        return replace(r, **over)

    # BOND：P 負が主駆動（快が無いほど募る）
    assert g_d(m(p=0.02)).bond > g_d(m(p=0.90)).bond
    # BOND：Pn 正も駆動する（不快でも募る）
    assert g_d(m(pn=0.80)).bond > g_d(r).bond
    # SEEKING：A 主駆動（高ぶりで探索）
    assert g_d(m(a=0.90)).seeking > g_d(m(a=0.10)).seeking
    # SAFETY：Dom 負が主駆動（無力・コントロール喪失で安全希求）
    assert g_d(m(dom=0.10)).safety > g_d(m(dom=0.90)).safety
    # REST：P/Pn/A すべて負（情動が鎮まると休みたい）
    assert g_d(m(p=0.02, pn=0.02, a=0.10)).rest > g_d(m(p=0.90, pn=0.90, a=0.90)).rest


# ── 蓄積：1 tick 増分＝rate·mult·learn·g·dt ──────────────────────────────────

def test_accumulate_one_tick_increment():
    d = accumulate(AiDrivers(), MoodPAD())  # 平静 → g=b_i
    step = _CFG.rate * _CFG.mult * _CFG.learn * _CFG.p_t
    assert d.seeking == pytest.approx(step * _CFG.bias_seeking, rel=1e-9)


def test_g1_reaches_threshold_in_one_minute():
    """設計不変：g_D=1・mult=learn=1 なら 120 tick（1分）で Θ_fire に達する。"""
    # 120·rate·P_T ≈ Θ_fire（発火mood §2.1／課題5 B の導出）
    assert 120 * _CFG.rate * _CFG.p_t == pytest.approx(_CFG.theta_fire, abs=1e-3)


def test_accumulate_clips_to_one():
    hot = AiDrivers(seeking=0.999)
    d = accumulate(hot, MoodPAD(a=0.99))  # さらに積んでも [0,1]
    assert d.seeking <= 1.0


# ── 発火・放電 ───────────────────────────────────────────────────────────────

def test_fired_at_threshold():
    f = fired(AiDrivers(seeking=_CFG.theta_fire, bond=0.5))
    assert f.seeking is True
    assert f.bond is False
    assert f.any is True
    assert fired(AiDrivers()).any is False


def test_discharge_fired_drive_to_near_zero():
    d = AiDrivers(seeking=1.0, bond=0.7)
    f = fired(AiDrivers(seeking=1.0))  # seeking だけ発火
    out = discharge(d, f)
    assert out.seeking == pytest.approx(0.0, abs=_CFG.epsilon + 1e-9)  # ~0 へ全放電
    assert out.bond == pytest.approx(0.7)  # 発火していない軸は不変


# ── 1 tick（蓄積→発火→放電）＝ループ接続 Slice 2a が呼ぶ純関数 ──────────────

def test_tick_below_threshold_accumulates_without_firing():
    d, f = tick(AiDrivers(), MoodPAD(), dt=_CFG.p_t)
    assert f.any is False
    assert d.seeking > 0.0  # 少し溜まった


def test_tick_at_threshold_fires_and_discharges():
    near = AiDrivers(seeking=_CFG.theta_fire)  # 閾値ちょうど
    d, f = tick(near, MoodPAD(a=0.99), dt=_CFG.p_t)  # さらに積んで発火
    assert f.seeking is True
    assert d.seeking == pytest.approx(0.0, abs=_CFG.epsilon + 1e-3)  # 放電で ~0


# ── 実 DB：ループ接続 Slice 2a の永続化経路（load→tick→save→load） ───────────

def test_tick_persists_to_drive5():
    import os

    import psycopg2

    from familiar_agent.drive_register import load_drives, save_drives

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.autocommit = True
    d0 = load_drives(conn)  # 空スタート（clean_db で agent_state truncate 済み）＝全0
    d1, _ = tick(d0, MoodPAD(), dt=1.0)
    save_drives(conn, d1)
    d2 = load_drives(conn)
    conn.close()
    assert d2.seeking == pytest.approx(d1.seeking)
    assert d2.seeking > 0.0  # 蓄積が永続化された
