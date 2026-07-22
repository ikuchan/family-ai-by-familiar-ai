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
    n = compute_n_pad([(ecur, 1.0)], self_pad=MoodPAD(), self_weight=0.5)
    # N_PAD.p = (1.0*0.9 + 0.5*0.5)/(1.0+0.5) = 0.7667
    assert n.p > 0.7
    # 旧 weight 2.0 なら (0.9+1.0)/3.0=0.633 で 0.7 未満だった
    old = compute_n_pad([(ecur, 1.0)], self_pad=MoodPAD(), self_weight=2.0)
    assert old.p < 0.7


def test_self_mi_emotion_roundtrip_and_default_neutral():
    """load/save の往復。未設定なら中立を返す（REST 書換口＝save）。"""
    from familiar_agent.db import get_db
    from familiar_agent.mood_register import (
        SELF_MI_STATE_KEY,
        load_self_mi_emotion,
        save_self_mi_emotion,
    )

    db = get_db()
    with db.lock:
        conn = db.conn()
        # 未設定 → 中立
        with conn.cursor() as cur:
            cur.execute("DELETE FROM agent_state WHERE state_key=%s", (SELF_MI_STATE_KEY,))
        conn.commit()
        assert load_self_mi_emotion(conn) == MoodPAD()

        # save → load 往復
        save_self_mi_emotion(conn, MoodPAD(0.7, 0.3, 0.5, 0.6))
        conn.commit()
        assert load_self_mi_emotion(conn) == MoodPAD(0.7, 0.3, 0.5, 0.6)

        # 後続テストへ漏らさないよう戻す
        with conn.cursor() as cur:
            cur.execute("DELETE FROM agent_state WHERE state_key=%s", (SELF_MI_STATE_KEY,))
        conn.commit()
