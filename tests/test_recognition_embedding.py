"""認識の InsightFace/ECAPA 載せ替え：埋め込み判定・保存・per-source 切替の検証。

実モデル（InsightFace/ECAPA）は重く GPU 依存なので、ここでは埋め込みの抽出をモックし、
純ロジック（cosine 最大＋しきい値）・保存の往復・RecognitionHint の形・自動切替の
per-source しきい値だけを見る。実モデル統合は実機確認に回す。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# ── best_match（純関数） ─────────────────────────────────────────────────────

def test_best_match_returns_closest_above_threshold():
    from familiar_agent.recognition.embedding_store import best_match

    enrolled = {
        "alice": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "bob": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    m = best_match(np.array([0.9, 0.1, 0.0], dtype=np.float32), enrolled, 0.35)
    assert m is not None
    key, score = m
    assert key == "alice"
    assert score > 0.9


def test_best_match_below_threshold_returns_none():
    from familiar_agent.recognition.embedding_store import best_match

    enrolled = {"alice": np.array([1.0, 0.0, 0.0], dtype=np.float32)}
    # 直交 → cosine 0 < 0.35
    assert best_match(np.array([0.0, 1.0, 0.0], dtype=np.float32), enrolled, 0.35) is None


def test_best_match_empty_or_zero_returns_none():
    from familiar_agent.recognition.embedding_store import best_match

    assert best_match(np.array([1.0, 0.0], dtype=np.float32), {}, 0.35) is None
    enrolled = {"alice": np.array([1.0, 0.0], dtype=np.float32)}
    assert best_match(np.array([0.0, 0.0], dtype=np.float32), enrolled, 0.35) is None


# ── EmbeddingStore（保存の往復） ─────────────────────────────────────────────

def test_embedding_store_roundtrip(tmp_path):
    from familiar_agent.recognition.embedding_store import EmbeddingStore

    path = tmp_path / "emb.pkl"
    store = EmbeddingStore(path)
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    store.save_embedding("alice", vec)

    reloaded = EmbeddingStore(path)
    got = reloaded.get()
    assert "alice" in got
    assert np.allclose(got["alice"], vec)


# ── RecognitionConfig（既定値） ──────────────────────────────────────────────

def test_recognition_config_defaults(monkeypatch):
    for k in (
        "FACE_THRESHOLD", "VOICE_THRESHOLD",
        "FACE_SWITCH_THRESHOLD", "VOICE_SWITCH_THRESHOLD",
    ):
        monkeypatch.delenv(k, raising=False)
    from familiar_agent.config import RecognitionConfig

    cfg = RecognitionConfig()
    assert cfg.face_threshold == pytest.approx(0.35)
    assert cfg.voice_threshold == pytest.approx(0.25)
    assert cfg.face_switch_threshold == pytest.approx(0.45)
    assert cfg.voice_switch_threshold == pytest.approx(0.35)


# ── recognize_face_async（モデルをモック） ──────────────────────────────────

@pytest.mark.asyncio
async def test_recognize_face_returns_hint_for_known_person(tmp_path):
    from familiar_agent.config import RecognitionConfig
    from familiar_agent.recognition import face as face_mod
    from familiar_agent.recognition.embedding_store import EmbeddingStore

    store = EmbeddingStore(tmp_path / "faces.pkl")
    store.save_embedding("alice", np.array([1.0, 0.0, 0.0], dtype=np.float32))

    manager = MagicMock()
    manager.list_persons.return_value = [{"id": "pid-alice", "name": "alice"}]

    with patch.object(
        face_mod, "_extract_face_embedding",
        return_value=np.array([0.95, 0.05, 0.0], dtype=np.float32),
    ):
        hint = await face_mod.recognize_face_async(
            "/tmp/x.jpg", manager, cfg=RecognitionConfig(), store=store
        )
    assert hint is not None
    assert hint.person_id == "pid-alice"
    assert hint.source == "face"
    assert 0.0 <= hint.confidence <= 1.0


@pytest.mark.asyncio
async def test_recognize_face_none_when_no_embedding(tmp_path):
    from familiar_agent.recognition import face as face_mod
    from familiar_agent.recognition.embedding_store import EmbeddingStore

    store = EmbeddingStore(tmp_path / "faces.pkl")
    manager = MagicMock()
    manager.list_persons.return_value = []
    with patch.object(face_mod, "_extract_face_embedding", return_value=None):
        hint = await face_mod.recognize_face_async("/tmp/x.jpg", manager, store=store)
    assert hint is None


# ── VoiceIdentifier（モデルをモック） ───────────────────────────────────────

@pytest.mark.asyncio
async def test_voice_identify_returns_hint_for_known_person(tmp_path):
    from familiar_agent.recognition import voice as voice_mod
    from familiar_agent.recognition.embedding_store import EmbeddingStore

    store = EmbeddingStore(tmp_path / "voices.pkl")
    store.save_embedding("pid-bob", np.array([0.0, 1.0, 0.0], dtype=np.float32))

    vi = voice_mod.VoiceIdentifier(MagicMock(), store=store)
    with patch.object(
        voice_mod, "_extract_voice_embedding",
        return_value=np.array([0.0, 0.98, 0.02], dtype=np.float32),
    ):
        hint = await vi.identify_async("/tmp/x.wav")
    assert hint is not None
    assert hint.person_id == "pid-bob"
    assert hint.source == "voice"


# ── apply_hint（per-source 自動切替しきい値） ───────────────────────────────

@pytest.mark.asyncio
async def test_apply_hint_uses_per_source_switch_threshold():
    """source 別しきい値で set_speaker を呼ぶか決める（呼び出し有無で見る）。

    返り値ではなく set_speaker の呼び出しで判定するのは、部屋が空のとき
    person_arrived が先に話者を立てるため、返り値が「切替の有無」を表さないため。
    """
    from familiar_agent.person_memory_manager import PersonMemoryManager, RecognitionHint

    thresholds = {"face": 0.45, "voice": 0.35}

    pmm = PersonMemoryManager(MagicMock(), switch_thresholds=thresholds)
    pmm.set_speaker = AsyncMock(return_value=True)  # type: ignore[method-assign]
    await pmm.apply_hint(RecognitionHint(person_id="p1", confidence=0.50, source="face"))
    pmm.set_speaker.assert_awaited_once()  # face 0.50 >= 0.45

    pmm2 = PersonMemoryManager(MagicMock(), switch_thresholds=thresholds)
    pmm2.set_speaker = AsyncMock(return_value=True)  # type: ignore[method-assign]
    await pmm2.apply_hint(RecognitionHint(person_id="p2", confidence=0.30, source="voice"))
    pmm2.set_speaker.assert_not_awaited()  # voice 0.30 < 0.35
