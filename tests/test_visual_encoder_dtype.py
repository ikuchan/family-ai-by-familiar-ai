"""出-dr：Hugging Face のモデルの読み込みと torch の既定の型。

2026-11-14 21:26 の起動だけ、見えのエンコーダ（DINOv2）が読み込みの 7 秒後から「expected scalar type Float but found
BFloat16」（層の正規化 `layer_norm`）で 17 分落ち続けた。`transformers` は読み込みのあいだ torch の既定の型を変えて戻す。
既定の型はプロセスに 1 つなので、Ruri（bf16）と同時に読むと出入りが重なり、既定が bf16 のまま残りうる。読み込みを
1 つずつにし、読み終わりに既定の型を確かめ、DINOv2 は型を指定して読む。落ちたら重みの型を記録に残す。
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import torch

from familiar_agent.core import model_resource as mr
from familiar_agent.recognition import visual_encoder as ve


def test_hugging_face_models_are_loaded_one_at_a_time(monkeypatch) -> None:
    monkeypatch.setattr(mr, "local_place", lambda name: name)
    inside = {"now": 0, "max": 0}
    guard = threading.Lock()

    def load(name: str) -> str:
        with guard:
            inside["now"] += 1
            inside["max"] = max(inside["max"], inside["now"])
        time.sleep(0.05)
        with guard:
            inside["now"] -= 1
        return name

    threads = [threading.Thread(target=mr.load_local_first, args=(n, load)) for n in "abc"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert inside["max"] == 1


def test_a_default_type_left_at_bf16_is_put_back(monkeypatch, caplog) -> None:
    monkeypatch.setattr(mr, "local_place", lambda name: name)

    def load(name: str) -> str:
        torch.set_default_dtype(torch.bfloat16)  # 出入りが重なって残った形
        return name

    try:
        with caplog.at_level(logging.WARNING, logger="familiar_agent.core.model_resource"):
            mr.load_local_first("org/model", load)
        assert torch.get_default_dtype() == torch.float32
        assert any("float32 に戻した" in r.getMessage() for r in caplog.records)
    finally:
        torch.set_default_dtype(torch.float32)


def test_the_visual_encoder_asks_for_float32(monkeypatch) -> None:
    import transformers

    seen: dict = {}

    class _Model(torch.nn.Module):
        pass

    def fake_model(place, **kw):
        seen.update(kw)
        return _Model()

    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", lambda p, **k: object())
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", fake_model)
    monkeypatch.setattr(ve, "load_local_first", lambda name, load: load("p"))
    monkeypatch.setattr(ve.VisualEncoder, "_torch_device", property(lambda self: "cpu"))
    ve.VisualEncoder()._load()
    assert seen.get("dtype") == torch.float32


def test_a_failed_embedding_logs_the_weight_types(monkeypatch, caplog) -> None:
    class _Mixed(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = torch.nn.LayerNorm(4)
            self.norm.weight.data = self.norm.weight.data.to(torch.bfloat16)

    enc = ve.VisualEncoder()
    model = _Mixed()
    monkeypatch.setattr(enc, "ensure", lambda: model)

    def boom(path):
        raise RuntimeError("expected scalar type Float but found BFloat16")

    monkeypatch.setattr(enc, "_embed_sync", boom)
    with caplog.at_level(logging.ERROR, logger="familiar_agent.recognition.visual_encoder"):
        assert asyncio.run(enc.embed("x.jpg")) is None
    text = caplog.records[-1].getMessage()
    assert "bfloat16 1" in text and "float32 1" in text and "既定の型 " in text
