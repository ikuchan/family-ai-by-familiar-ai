"""問い用の埋め込みは、同じ文の文書用の控えがあればそれを使う（出-bz-い・2026-10-13・本人の決定 A）。

版を書くときに版の文を埋め込み（文書用）、直後の完了の想起で同じ文を手がかりとしてもう一度埋め込んでいた（問い用）。
置き場は分かれているが、どちらも同じモデルを同じ設定で呼ぶので、同じベクトルを 2 回計算していた。10/13 00:35 に、
約 6,800 字の版を書くのに 0.82 秒、その版を手がかりにした想起に 2.08 秒かかった。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from familiar_agent.store.embedding import _EmbeddingModel


def _model():
    m = _EmbeddingModel()
    fake = MagicMock()
    fake.encode = MagicMock(
        side_effect=lambda texts, **kw: np.ones((len(texts), 4)) * len(texts[0])
    )
    m.ensure = lambda: fake  # type: ignore[method-assign]
    return m, fake


def test_a_query_reuses_the_vector_of_the_same_document():
    m, fake = _model()
    doc = m.encode_document(["版の文"])
    query = m.encode_query(["版の文"])
    assert query == doc
    assert fake.encode.call_count == 1


def test_a_different_query_is_encoded():
    """**反証の側**：違う文は、今までどおり問い用に計算する。"""
    m, fake = _model()
    m.encode_document(["版の文"])
    m.encode_query(["人の言葉"])
    assert fake.encode.call_count == 2


def test_both_encodings_use_the_same_settings():
    """控えを共有してよいのは、2 つが同じ設定でモデルを呼ぶから。設定が分かれたら、この試験が落ちる。"""
    m, fake = _model()
    m.encode_document(["a"])
    m.encode_query(["b"])
    (_, doc_kw), (_, query_kw) = [(c.args, c.kwargs) for c in fake.encode.call_args_list]
    assert doc_kw == query_kw
