"""在席者相関 p（想起の第5軸・役割2・課題5 v0.26／[D-在席相関]）。

在席他者 q ごとに q 視点の situated コサインを r と同じ伸長で r_{p,q} 化し、
noisy-OR p = 1 − Π_q(1 − r_{p,q}) で束ねる（自分・話者は呼び出し側で除外）。
在席他者ゼロなら p 項を分母ごと外す（挙動不変）。スライス1＝score 軸のみ。
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from familiar_agent.tools.memory import _score_breakdown


def _bd(**kw):
    base = dict(
        cosine=0.8,
        ts=datetime.now(timezone.utc),
        last_recalled_at=None,
        recall_count=0,
        groundedness_g0=0.4,
        groundedness_n=0,
        half_life_days=30.0,
        floor=0.1,
        w_t=1.0,
        w_e=1.0,
        w_g=1.5,
    )
    base.update(kw)
    return _score_breakdown(**base)


# ── _score_breakdown への p/w_p 追加 ─────────────────────────────────────────

def test_score_breakdown_default_has_no_p():
    """p を渡さなければ従来どおり（p 項なし・後方互換）。"""
    parts = _bd()
    assert parts.p is None


def test_score_breakdown_p_raises_m_when_high():
    """在席者相関 p が高いと M（加算部）が上がる。"""
    without = _bd()
    withp = _bd(p=1.0, w_p=1.0)
    assert withp.p == pytest.approx(1.0)
    assert withp.m > without.m


def test_score_breakdown_wp_zero_is_unchanged():
    """w_p=0 なら p を渡しても M は不変（項を外す）。"""
    base = _bd()
    zero = _bd(p=1.0, w_p=0.0)
    assert zero.m == pytest.approx(base.m)


# ── 束ね（noisy-OR）と伸長：facade _presence_correlation ─────────────────────

def _facade_for_corr(cosines_by_person: dict[str, dict[str, float]]):
    """_presence_correlation を DB なしで回すための最小 facade スタブ。

    cosines_by_person[q][obs_id] = situated コサイン。
    """
    from familiar_agent.tools.memory import ObservationMemory

    mem = ObservationMemory.__new__(ObservationMemory)
    situated = MagicMock()
    situated._embedding_mu.return_value = None
    situated._get_perspective_vec.return_value = None

    def _situate_dummy(*a, **k):
        return None

    obs = MagicMock()
    obs.situated_cosines.side_effect = (
        lambda q_sql, obs_ids, person_id: cosines_by_person.get(person_id, {})
    )
    mem._situated = situated
    mem._observations = obs
    return mem


def test_presence_correlation_noisy_or_over_present_others(monkeypatch):
    import familiar_agent.tools.memory as m

    import numpy as np

    # situate はダミー（コサインはスタブが直接返す）。伸長は c_lo=0/c_hi=1 で恒等。
    monkeypatch.setattr(m, "_situated_vector", lambda *a, **k: np.zeros(3, dtype=np.float32))
    monkeypatch.setattr(m, "vec_to_sql", lambda v: "q")

    mem = _facade_for_corr({
        "alice": {"o1": 0.5, "o2": 0.0},
        "bob":   {"o1": 0.5, "o2": 0.0},
    })
    p = mem._presence_correlation(
        None, ["o1", "o2"], ["alice", "bob"], c_lo=0.0, c_hi=1.0,
    )
    # o1: noisy-OR(0.5, 0.5) = 1 − 0.5·0.5 = 0.75
    assert p["o1"] == pytest.approx(0.75)
    # o2: 両者 0 → p = 0
    assert p["o2"] == pytest.approx(0.0)


def test_presence_correlation_empty_present_others_returns_empty():
    mem = _facade_for_corr({})
    assert mem._presence_correlation(None, ["o1"], [], c_lo=0.0, c_hi=1.0) == {}
    assert mem._presence_correlation(None, [], ["alice"], c_lo=0.0, c_hi=1.0) == {}


def test_config_recall_w_p_default(monkeypatch):
    monkeypatch.delenv("RECALL_W_P", raising=False)
    from familiar_agent.config import MemoryConfig

    assert MemoryConfig().recall_w_p == pytest.approx(1.0)


def test_config_recall_presence_expand_default(monkeypatch):
    monkeypatch.delenv("RECALL_PRESENCE_EXPAND", raising=False)
    from familiar_agent.config import MemoryConfig

    assert MemoryConfig().recall_presence_expand is True


# ── slice-2：候補集合拡張（在席他者視点で候補を union） ──────────────────────

def _row(oid: str, score: float) -> dict:
    return {
        "id": oid, "content": f"content-{oid}", "timestamp": "2026-07-01T10:00:00+09:00",
        "direction": "in", "kind": "observation", "emotion": "neutral", "image_path": None,
        "groundedness_g0": 1.0, "groundedness_n": 0, "recall_count": 0, "last_recalled_at": None,
        "emotion_p": 0.5, "emotion_pn": 0.5, "emotion_a": 0.5, "emotion_dom": 0.5,
        "score": score,
    }


def _recall_facade(by_vector_seq, situated_cosines_fn, person_id="spk"):
    from familiar_agent.tools.memory import ObservationMemory

    mem = ObservationMemory.__new__(ObservationMemory)
    mem._person_id = person_id
    emb = MagicMock()
    emb.encode_query.return_value = [[1.0, 0.0, 0.0]]
    mem._embedder = emb
    sit = MagicMock()
    sit._embedding_mu.return_value = None
    sit._get_perspective_vec.return_value = None
    mem._situated = sit
    obs = MagicMock()
    seq = list(by_vector_seq)
    calls = {"i": 0}

    def _bv(q_sql, n, kind=None, exclude_ids=None):
        i = calls["i"]
        calls["i"] += 1
        return seq[i] if i < len(seq) else []

    obs.by_vector.side_effect = _bv
    obs.situated_cosines.side_effect = situated_cosines_fn
    mem._observations = obs
    return mem


def _patch_recall_env(monkeypatch):
    import numpy as np

    import familiar_agent.tools.memory as m
    from familiar_agent.mood_register import MoodPAD

    monkeypatch.setattr(m, "_situated_vector", lambda *a, **k: np.zeros(3, dtype=np.float32))
    monkeypatch.setattr(m, "vec_to_sql", lambda v: "q")
    monkeypatch.setattr(m, "load_current_mood", lambda: MoodPAD())


def _sc(q_sql, obs_ids, person_id):
    # 話者視点＝r 補完（B のみ）。在席他者 q1 視点＝B が強く結びつく。
    # 実 situated_cosines は要求 obs_id ぶんしか返さない（WHERE obs_id = ANY）ので絞る。
    full = {"B": 0.15} if person_id == "spk" else {"A": 0.0, "B": 0.9}
    return {k: v for k, v in full.items() if k in obs_ids}


def test_recall_slice2_expands_candidate_from_present_other(monkeypatch):
    monkeypatch.delenv("RECALL_PRESENCE_EXPAND", raising=False)  # 既定 on
    _patch_recall_env(monkeypatch)
    # 話者候補＝A のみ。在席他者 q1 視点の候補＝B（話者クエリと無関係でも W へ）。
    mem = _recall_facade([[_row("A", 0.8)], [_row("B", 0.7)]], _sc)
    res = mem.recall("q", n=5, present_others=["q1"])
    ids = {r["memory_id"] for r in res}
    assert "A" in ids
    assert "B" in ids  # 在席他者視点で候補集合に入った


def test_recall_slice2_toggle_off_is_slice1(monkeypatch):
    monkeypatch.setenv("RECALL_PRESENCE_EXPAND", "false")  # 退避＝slice-1 のみ
    _patch_recall_env(monkeypatch)
    # 拡張オフなら在席他者視点の by_vector は呼ばれず、B は候補に入らない。
    mem = _recall_facade([[_row("A", 0.8)]], _sc)
    res = mem.recall("q", n=5, present_others=["q1"])
    ids = {r["memory_id"] for r in res}
    assert ids == {"A"}


# ── 実 DB：p が想起スコアへ効く（在席他者ありでスコアが上がる） ────────────────

def test_recall_present_others_raises_score():
    import os
    from unittest.mock import patch

    import psycopg2

    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    ps = (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[1.0, 0.0, 0.0]]),
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]),
    )
    for p in ps:
        p.start()
    try:
        # 在席他者 q を persons へ登録（save 時に q 視点の situated 行が作られる）。
        c = psycopg2.connect(os.environ["DATABASE_URL"])
        c.autocommit = True
        with c.cursor() as cur:
            cur.execute(
                "INSERT INTO persons (id, name, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                ("q-person", "Qperson", "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
            )
        c.close()

        mem = ObservationMemory()
        mem.save("presence corr target", kind="observation")

        base = mem.recall("presence corr target", n=1)
        boosted = mem.recall("presence corr target", n=1, present_others=["q-person"])
        assert base and boosted
        # 在席他者 q が memory に強く結びつく（同じ埋め込み）→ p>0 で M が上がる。
        assert boosted[0]["fit"] > base[0]["fit"]
    finally:
        for p in ps:
            p.stop()


def test_recall_no_present_others_unchanged():
    """present_others 無指定なら従来スコアと一致（回帰・p 項落ち）。"""
    from unittest.mock import patch

    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    ps = (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[1.0, 0.0, 0.0]]),
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]),
    )
    for p in ps:
        p.start()
    try:
        mem = ObservationMemory()
        mem.save("presence corr regression", kind="observation")
        a = mem.recall("presence corr regression", n=1)
        b = mem.recall("presence corr regression", n=1, present_others=[])
        assert a and b
        assert a[0]["fit"] == pytest.approx(b[0]["fit"])
    finally:
        for p in ps:
            p.stop()
