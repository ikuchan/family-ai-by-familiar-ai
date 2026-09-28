"""自己認識 MI の emotion を Config 既定＋REST 書換可な store へ、重みを Config 0.5 へ。

自己 MI の emotion は W が空のときのデフォルト感情として使い、外部 MI が入れば重み0.5の
一員として N_PAD に参加する（支配しない）。emotion は agent_state に置き REST が書き換え
られる（既定は中立）。
"""

from __future__ import annotations

import pytest

from familiar_agent.mood_register import MoodPAD


def test_config_self_mi_weight_default(monkeypatch):
    monkeypatch.delenv("SELF_MI_WEIGHT", raising=False)
    from familiar_agent.config import MemoryConfig

    assert MemoryConfig().self_mi_weight == pytest.approx(0.5)


def test_compute_n_pad_empty_returns_self_pad():
    """外部 MI が無ければ N_PAD は自己 MI emotion（デフォルト感情の役）。"""
    from familiar_agent.mood_register import compute_n_pad

    sp = MoodPAD(0.8, 0.2, 0.6, 0.7)
    assert compute_n_pad([], self_pad=sp, self_weight=0.5) == sp


def test_compute_n_pad_light_self_weight_lets_ecur_move():
    """重み0.5なら現ターン感情 E_cur(重み1.0)が N_PAD を動かせる（旧2.0では潰れていた）。"""
    from familiar_agent.mood_register import compute_n_pad

    ecur = MoodPAD(0.9, 0.1, 0.8, 0.6)
    light = compute_n_pad([(ecur, 1.0)], self_pad=MoodPAD(), self_weight=0.5)
    heavy = compute_n_pad([(ecur, 1.0)], self_pad=MoodPAD(), self_weight=2.0)
    # 自己認識 MI の既定は (0.10, 0.10, 0.50, 0.50)（案A）。錨が E_cur の 0.9 より下に
    # あるので、重みが増えるほど N_PAD は引き下げられる。
    #   light = (0.9 + 0.5*0.10)/1.5 = 0.6333    heavy = (0.9 + 2.0*0.10)/3.0 = 0.3667
    # 絶対値の閾値でなく大小で見る。錨の位置を動かしてもこの関係は変わらない。
    assert light.p > heavy.p
    # 軽い側では E_cur が錨より現ターン寄りに N_PAD を保てる（中点より上に残る）。
    assert light.p > 0.5
