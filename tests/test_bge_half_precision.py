"""環-ay：bge-m3 を GPU では float16 で読む（約 1.07 GB 空く）。CPU は float32 のまま。

測った（2026-11-21、本番の記録の本文 5,950 種、読み取りだけ）：float16 で nan になる本文は 0 種、float32 との点の近さは
最小 0.99972。146 問の上位 7 件は、記録ごとに厳密に比べて平均 6.84 件が重なった。
"""

from __future__ import annotations

import torch

from familiar_agent.store import embedding as emb


def _seen(monkeypatch, device: str) -> dict:
    import sentence_transformers

    seen: dict = {}

    def fake(place, **kw):
        seen.update(kw)
        return object()

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fake)
    monkeypatch.setattr(emb, "load_local_first", lambda name, load: load(name))
    monkeypatch.setenv("EMBEDDING_DEVICE", device)
    emb._EmbeddingModel()._load()
    return seen


def test_bge_m3_is_loaded_in_float16_on_the_gpu(monkeypatch) -> None:
    seen = _seen(monkeypatch, "cuda")
    assert seen.get("model_kwargs") == {"dtype": torch.float16}
    assert seen.get("device") == "cuda"


def test_bge_m3_stays_float32_on_the_cpu(monkeypatch) -> None:
    seen = _seen(monkeypatch, "cpu")
    assert not seen.get("model_kwargs")


def test_with_no_device_set_the_gpu_still_gets_float16(monkeypatch) -> None:
    """家の機体の形：載せる先の指定は無く（ライブラリに任せる）、GPU がある。"""
    import sentence_transformers

    seen: dict = {}

    def fake(place, **kw):
        seen.update(kw)
        return object()

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fake)
    monkeypatch.setattr(emb, "load_local_first", lambda name, load: load(name))
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    emb._EmbeddingModel()._load()
    assert seen.get("device") is None
    assert seen.get("model_kwargs") == {"dtype": torch.float16}


def test_a_vector_that_is_not_a_number_becomes_zero_and_is_not_cached(monkeypatch) -> None:
    import numpy as np

    class _Model:
        def encode(self, texts, **kw):
            assert kw.get("normalize_embeddings") is False  # 長さは float32 でそろえる
            out = np.ones((len(texts), 4), dtype=np.float16) * 3
            out[0, 0] = np.nan
            return out

    m = emb._EmbeddingModel()
    monkeypatch.setattr(m, "ensure", lambda: _Model())
    vecs = m.encode_document(["溢れた文", "ふつうの文"])
    assert vecs[0] == [0.0, 0.0, 0.0, 0.0]
    assert abs(float(np.linalg.norm(vecs[1])) - 1.0) < 1e-6
    assert m._key("溢れた文") not in m._d_cache and m._key("ふつうの文") in m._d_cache
