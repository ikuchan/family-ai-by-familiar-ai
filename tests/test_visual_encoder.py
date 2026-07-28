"""見えの埋め込み（DINOv2）。

`知覚在席` §3-2 の「見えの普通／変化」を担う視覚エンコーダ。テキストを介さず構造の変化に
敏感で、CLIP を採らない理由もそこにある（`用語一覧`）。`facebook/dinov2-small`（ViT-S/14・
384次元・Apache-2.0）を使い、新しい依存は増やさない（`transformers` は導入済み）。

在席の確認と同じ常駐から呼ばれるので、**例外を投げない**。読めないときは `None` を返し、
その回の「普通」の更新を見送る。
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from familiar_agent.recognition.visual_encoder import VisualEncoder


def _encoder(vector=None):
    e = VisualEncoder()
    e._model = MagicMock()
    e._processor = MagicMock()
    e._embed_sync = MagicMock(return_value=vector)
    return e


def test_an_image_becomes_a_vector():
    e = _encoder(vector=[0.1] * 384)
    got = asyncio.run(e.embed("/tmp/a.jpg"))
    assert got is not None and len(got) == 384


def test_encoding_runs_off_the_event_loop():
    """GPU の重い呼び出しでイベントループを止めない（`コード規約` の並行処理）。"""
    e = _encoder(vector=[0.1] * 384)
    with patch("asyncio.to_thread", wraps=asyncio.to_thread) as spy:
        asyncio.run(e.embed("/tmp/a.jpg"))
    assert spy.called


def test_a_model_that_cannot_load_yields_nothing_rather_than_raising():
    e = VisualEncoder()
    with patch("transformers.AutoModel.from_pretrained",
               MagicMock(side_effect=RuntimeError("no weights"))):
        assert asyncio.run(e.embed("/tmp/a.jpg")) is None


def test_a_failed_encoding_yields_nothing():
    e = VisualEncoder()
    e._model = MagicMock()
    e._processor = MagicMock()
    e._embed_sync = MagicMock(side_effect=RuntimeError("cuda oom"))
    assert asyncio.run(e.embed("/tmp/a.jpg")) is None


def test_the_model_is_loaded_once_and_reused():
    e = VisualEncoder()
    made = MagicMock(return_value=MagicMock())
    with patch("transformers.AutoModel.from_pretrained", made), \
         patch("transformers.AutoImageProcessor.from_pretrained", MagicMock()):
        e._embed_sync = MagicMock(return_value=[0.0] * 384)
        asyncio.run(e.embed("/tmp/a.jpg"))
        asyncio.run(e.embed("/tmp/b.jpg"))
    assert made.call_count <= 1
