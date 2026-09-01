"""概念6「適合度」（fit・記号 f）と概念4「地力」（merit・記号 m）。

想起の採点には、名前の無いまま2つの合成量があった。

- **地力**：`(w_t·t + w_e·e + w_g·g + w_p·p) / Σw`。時間経過を含んだ重要度で、関連を
  含まない。記号 `m` はそのまま使う。
- **適合度**：`r^(w_r) × 地力`。**W を選ぶのはこれ**で、いまの問いへの適合を含む。

`score` という語も別の量に相乗りしていた。store の `by_vector` が返す行の `score` は
**生コサイン**、`DecayState.score` は**時間減衰**である。返り値の最終値だけを `fit` に
することで、どれを指しているかが名前で決まる。

ここでは (1) 内訳の型が `fit` と `m` を持つこと、(2) 返り値のキーが `fit` であること、
(3) `fit = r^(w_r) × m` が成り立つこと、(4) 旧キーが残っていないこと、を見る。
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel, _score_breakdown


def _row(obs_id: str = "obs-1") -> dict:
    return {
        "id": obs_id, "content": "むかしの話", "timestamp": None,
        "last_recalled_at": None,
        "groundedness_g0": 1.0, "groundedness_n": 0,
        "emotion_p": 0.5, "emotion_pn": 0.5, "emotion_a": 0.5, "emotion_dom": 0.5,
        "direction": "発話", "kind": "observation", "emotion": "neutral",
        "image_path": None, "score": 0.5,
    }


def test_breakdown_exposes_fit_and_merit() -> None:
    """内訳は `fit`（適合度）と `m`（地力）を持ち、旧名 `score` は持たない。"""
    parts = _score_breakdown(
        0.5, None, None, 1.0, 0,
        half_life_days=3.0, floor=0.001,
    )
    assert hasattr(parts, "fit"), "適合度 fit が無い"
    assert hasattr(parts, "m"), "地力 m が無い"
    assert not hasattr(parts, "score"), "旧名 score が残っている"


def test_fit_is_relevance_gate_times_merit() -> None:
    """`fit = r^(w_r) × m`。関連ゲートは指数、地力は加重平均。"""
    parts = _score_breakdown(
        0.5, None, None, 1.0, 0,
        half_life_days=3.0, floor=0.001, w_r=2.0,
    )
    assert parts.fit == pytest.approx((parts.r ** 2.0) * parts.m)


def test_merit_excludes_relevance() -> None:
    """地力は関連を含まない（$w_r$ を変えても動かない）。"""
    a = _score_breakdown(0.5, None, None, 1.0, 0,
                         half_life_days=3.0, floor=0.001, w_r=1.0)
    b = _score_breakdown(0.5, None, None, 1.0, 0,
                         half_life_days=3.0, floor=0.001, w_r=3.0)
    assert a.m == pytest.approx(b.m), "地力が関連の重みで動いている"
    assert a.fit != pytest.approx(b.fit), "適合度が関連の重みで動いていない"


def test_recall_result_key_is_fit() -> None:
    """`recall()` が返す dict のキーは `fit`。旧キー `score` は無い。"""
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]),
    ):
        mem = ObservationMemory()
        with (
            patch.object(mem._observations, "by_vector", return_value=[_row()]),
            patch.object(mem._observations, "by_time", return_value=[]),
            patch.object(mem._observations, "by_emotion", return_value=[]),
            patch.object(mem._observations, "situated_cosines", return_value={"obs-1": 0.5}),
        ):
            got = mem.recall("q", n=7)

    assert got, "候補が採点まで届いていない（前提が崩れている）"
    assert "fit" in got[0], "返り値キーが fit になっていない"
    assert "score" not in got[0], "旧キー score が残っている"


def test_store_rows_keep_their_own_score() -> None:
    """store の `by_vector` が返す行の `score`（生コサイン）は変えない。

    採点前の素の値で、`recall()` の返り値とは別物である。ここまで改名すると、層を
    またいで意味が繋がっているように見えてしまう。
    """
    import inspect

    from familiar_agent.store.observations import ObservationStore

    src = inspect.getsource(ObservationStore.by_vector)
    assert "score" in src, "store 側の score まで消えている"
