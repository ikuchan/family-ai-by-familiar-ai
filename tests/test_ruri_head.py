"""記-y：Ruri を 2 つめの頭として、関連の値 r に混ぜる（2026-11-03・本人の決定）。

r は「頭ごとに 0〜1 へ伸ばした値の大きいほう」。どちらの頭が r を決めたかを `head` に残す
（ログで「どっちが選んだか」を出すため）。Ruri の値が無い・同じ値なら bge-m3 とする。
"""

from __future__ import annotations

import pytest

from familiar_agent.config import MemoryConfig
from familiar_agent.tools.memory import _score_breakdown

_BASE = dict(half_life_days=3.0, floor=0.001)


def _parts(cosine: float, ruri: "float | None", **kw):
    return _score_breakdown(
        cosine,
        None,
        None,
        1.0,
        0,
        ruri_cosine=ruri,
        ruri_c_lo=0.782,
        ruri_c_hi=0.995,
        **_BASE,
        **kw,
    )


def test_r_is_the_larger_stretched_head() -> None:
    """Ruri を伸ばした値が bge-m3 より大きければ、r は Ruri の値で、頭は Ruri。"""
    parts = _parts(0.10, 0.90)
    ruri_r = (0.90 - 0.782) / (0.995 - 0.782)
    assert parts.r == pytest.approx(ruri_r)
    assert parts.head == "Ruri"
    assert parts.r_ruri == pytest.approx(ruri_r)


def test_bge_wins_when_it_is_larger() -> None:
    parts = _parts(0.60, 0.80)
    assert parts.r == pytest.approx(0.60)
    assert parts.head == "bge-m3"


def test_tie_goes_to_bge() -> None:
    """同じ値なら bge-m3（いまの頭を基準にする）。"""
    # 両方とも上限で 1.0 に止まる＝浮動小数の丸めに左右されない、ちょうど同じ値。
    parts = _parts(1.0, 0.999)
    assert parts.r == 1.0 and parts.r_ruri == 1.0
    assert parts.head == "bge-m3"


def test_without_ruri_value_r_is_bge_only() -> None:
    """Ruri の数値が無い記録（休み・未計算）は、bge-m3 だけで r を決める。"""
    parts = _parts(0.30, None)
    assert parts.r == pytest.approx(0.30)
    assert parts.head == "bge-m3"
    assert parts.r_ruri is None


def test_ruri_is_clipped_to_unit_range() -> None:
    assert _parts(0.0, 0.70).r_ruri == 0.0
    assert _parts(0.0, 0.999).r_ruri == 1.0


def test_default_call_is_unchanged() -> None:
    """Ruri を渡さない呼び出しは、いままでと同じ r（bge-m3 だけ）になる。"""
    parts = _score_breakdown(0.4, None, None, 1.0, 0, **_BASE)
    assert parts.r == pytest.approx(0.4)
    assert parts.head == "bge-m3"


def test_config_defaults() -> None:
    """値は config.py の既定に置く（中央値と上位 0.1% で bge-m3 の目盛りに重ねた・2026-11-03）。"""
    cfg = MemoryConfig()
    assert cfg.ruri_model == "cl-nagoya/ruri-v3-310m"
    assert cfg.ruri_c_lo == pytest.approx(0.782)
    assert cfg.ruri_c_hi == pytest.approx(0.995)
    assert cfg.ruri_primary_n == 50


# ── A3：Ruri の入れ物（縮退の側・前置き・失敗は「無し」） ───────────────────


class _FakeST:
    """SentenceTransformer の代わり。渡された文を覚え、決まった数値を返す。"""

    def __init__(self, fail: bool = False, nan: bool = False) -> None:
        self.seen: list[str] = []
        self.fail = fail
        self.nan = nan

    def encode(self, texts, **kw):
        import numpy as np

        if self.fail:
            raise RuntimeError("encode broke")
        self.seen.extend(texts)
        out = np.ones((len(texts), 768), dtype=np.float32) / np.sqrt(768)
        if self.nan:
            out[0, 0] = np.nan
        return out


def _ruri_with(fake):
    from unittest.mock import patch

    from familiar_agent.store.embedding import _RuriModel

    m = _RuriModel()
    p = patch.object(_RuriModel, "_load", return_value=fake)
    p.start()
    return m, p


def test_ruri_adds_prefixes() -> None:
    """記録には「検索文書: 」、問いには「検索クエリ: 」を付ける（Ruri v3 の使い方）。"""
    fake = _FakeST()
    m, p = _ruri_with(fake)
    try:
        doc = m.encode_document(["本文"])
        q = m.encode_query(["問い"])
    finally:
        p.stop()
    assert fake.seen == ["検索文書: 本文", "検索クエリ: 問い"]
    assert doc is not None and len(doc[0]) == 768
    assert q is not None and len(q[0]) == 768


def test_ruri_failure_returns_none_not_zeros() -> None:
    """数値にできなければ None（0 を返すと「近さ 0」と区別できない値が残る）。"""
    m, p = _ruri_with(_FakeST(fail=True))
    try:
        assert m.encode_document(["本文"]) is None
        assert m.encode_query(["問い"]) is None
    finally:
        p.stop()


def test_ruri_nan_is_treated_as_failure() -> None:
    """数にならない値（fp16 で出た）が混じったら、その結果は使わない。"""
    m, p = _ruri_with(_FakeST(nan=True))
    try:
        assert m.encode_document(["本文"]) is None
    finally:
        p.stop()


def test_ruri_load_failure_degrades() -> None:
    """読み込めなくても例外を投げない（縮退の側）。以後は休み（None）。"""
    from unittest.mock import patch

    from familiar_agent.store.embedding import _RuriModel

    m = _RuriModel()
    with patch.object(_RuriModel, "_load", side_effect=RuntimeError("no gpu")):
        assert m.encode_query(["問い"]) is None
    assert m.failed


def test_ruri_precision_by_device() -> None:
    """GPU では bf16 で直接読み込む（fp16 は数にならない値が出た・fp32 で読むと一瞬メモリを食う）。CPU は fp32。"""
    import torch

    from familiar_agent.store.embedding import _ruri_model_kwargs

    assert _ruri_model_kwargs("cuda") == {"dtype": torch.bfloat16}
    assert _ruri_model_kwargs("cuda:0") == {"dtype": torch.bfloat16}
    assert _ruri_model_kwargs("cpu") == {}


def test_ruri_cosine_without_coefficients_is_refused() -> None:
    """係数を渡し忘れると、伸ばしていない Ruri（0.8 前後）が bge-m3 をいつも上回る。黙って通さない。"""
    with pytest.raises(ValueError):
        _score_breakdown(0.4, None, None, 1.0, 0, ruri_cosine=0.85, **_BASE)
