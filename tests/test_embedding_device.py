"""埋め込みモデルを載せる先を選べるようにする（`EMBEDDING_DEVICE`）。

`SentenceTransformer(model_name)` は device を指定していないので、GPU があれば自動で GPU を
使う。テストは `-n auto`（この機体では12）でワーカーを起こし、**ワーカーごとにモデルを GPU へ
載せる**。1プロセスあたり 1.7〜2.3 GiB を占め、RTX 3060（12 GiB）では6プロセスで使い切る。

実測のログ：

```
Embedding model load failed: CUDA out of memory. Tried to allocate 20.00 MiB.
GPU 0 has a total capacity of 11.63 GiB of which 5.81 MiB is free.
```

その結果、全体実行でときどきワーカーがプロセスごと落ちた（`worker 'gwN' crashed`）。どの
テストがどのワーカーへ割り当てられるかで載せるプロセス数が変わるため、再現が不安定だった。

**載せる先を CPU へ移しても解決しない。** 実測で CPU 版は 1プロセス +1140MiB を占め、
この機体の RAM 15GiB（swap 4GiB は常時ほぼ満杯）では 12ワーカーが同じく足りない。実際、
CPU へ移した実行でもワーカーが落ち、所要が 62秒から 182秒へ伸びた。

そこで**同時に載る数**を抑える（`scripts/run_tests.sh` の並列度）。載せる先を選べること
自体は、資源の違う機体で回すときに要るので残す。既定は自動（実機の挙動を変えない）。
"""

from __future__ import annotations

import pathlib
import re

from familiar_agent.store.embedding import _EmbeddingModel


def test_device_defaults_to_auto(monkeypatch) -> None:
    """既定は自動（従来どおり）。実機の挙動を変えない。"""
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    assert _EmbeddingModel(model_name="dummy").device is None


def test_device_can_be_pinned(monkeypatch) -> None:
    """`EMBEDDING_DEVICE` で載せる先を選べる。"""
    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
    assert _EmbeddingModel(model_name="dummy").device == "cpu"


def test_the_device_is_passed_to_the_model(monkeypatch) -> None:
    """選んだ先が、モデルの読み込みへ実際に渡る。"""
    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
    seen = {}

    class _FakeST:
        def __init__(self, name, device=None):
            seen["name"] = name
            seen["device"] = device

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeST)
    m = _EmbeddingModel(model_name="dummy")
    m._load()
    assert seen.get("device") == "cpu", f"device が渡っていない: {seen}"


def test_no_device_is_passed_when_auto(monkeypatch) -> None:
    """自動のときは device を渡さない（従来と同じ呼び方）。"""
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    seen = {}

    class _FakeST:
        def __init__(self, name, device=None):
            seen["name"] = name
            seen["device"] = device

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeST)
    m = _EmbeddingModel(model_name="dummy")
    m._load()
    assert seen.get("device") is None, f"自動なのに device を渡している: {seen}"


def test_the_test_parallelism_is_bounded() -> None:
    """テストの並列度は、資源から決めた固定値にする（`-n auto` にしない）。

    ワーカーごとに埋め込みモデルが載る。論理CPU数（この機体で12）だと VRAM も RAM も
    足りず、ワーカーがプロセスごと落ちる。CPU へ逃がしても RAM 側で同じことが起きる
    （1プロセス 1140MiB × 12 に対し RAM 15GiB・swap は常時ほぼ満杯）ので、
    載せる先を変えるのではなく**同時に載る数**を抑える。
    """
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_tests.sh"
    text = script.read_text(encoding="utf-8")
    assert "-n auto" not in text, "並列度が論理CPU数のままになっている"
    assert re.search(r'RUN_TESTS_PARALLEL:--n \d+', text), (
        "並列度が固定値で指定されていない"
    )
