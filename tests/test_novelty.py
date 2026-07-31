"""取込 novelty（内容の新規性）→ a0（課題5 v0.26）。

AGENT_SELF 視点・self_model 除外・situated 近傍 K 件の平均コサインの裏返し（1−平均）。
保存時に a0 = clip(w_n·novelty, 0, cap) を設定する。
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import psycopg2
import pytest

from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel


# ── content_novelty のロジック（conn をモック・DB 不要） ─────────────────────

class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a):
        pass

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _FakeCursor(self._rows)


def _store_with_cosines(cosines):
    from familiar_agent.store.observations import ObservationStore

    store = ObservationStore.__new__(ObservationStore)
    store._situated = MagicMock()
    store._situated.situate.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    conn = _FakeConn([{"c": c} for c in cosines])
    return store, conn


def test_content_novelty_high_when_far():
    store, conn = _store_with_cosines([0.0] * 7)  # 全部直交 → 平均0 → novelty 1
    assert store.content_novelty(np.array([1.0, 0.0, 0.0]), conn, k=7, default=0.5) == 1.0


def test_content_novelty_low_when_near():
    store, conn = _store_with_cosines([1.0] * 7)  # 全部一致 → 平均1 → novelty 0
    assert store.content_novelty(np.array([1.0, 0.0, 0.0]), conn, k=7, default=0.5) == 0.0


def test_content_novelty_default_when_few_neighbors():
    store, conn = _store_with_cosines([0.9] * 3)  # K 未満 → 既定
    assert store.content_novelty(np.array([1.0, 0.0, 0.0]), conn, k=7, default=0.5) == 0.5


# ── a0 への配線（実 DB・autouse truncate で空スタート） ─────────────────────

def _a0(content: str) -> float | None:
    c = psycopg2.connect(os.environ["DATABASE_URL"])
    c.autocommit = True
    with c.cursor() as cur:
        cur.execute("SELECT groundedness_g0 FROM observations WHERE content=%s", (content,))
        row = cur.fetchone()
    c.close()
    return float(row[0]) if row else None


def _fixed_embed():
    return (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[1.0, 0.0, 0.0]]),
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0, 0.0, 0.0]]),
    )


def test_novelty_wires_a0_on_save():
    ps = _fixed_embed()
    for p in ps:
        p.start()
    try:
        mem = ObservationMemory()
        # 空ストア → 近傍0<7 → 既定 novelty 0.5 → a0 = 1.5*0.5 = 0.75
        mem.save("novelty first unique", kind="observation")
        assert _a0("novelty first unique") == pytest.approx(0.75, abs=0.02)

        # 同じ埋め込みの記憶を溜める（内容は別で dedup 回避）
        for i in range(8):
            mem.save(f"novelty filler {i}", kind="observation")
        # 近傍が K 件以上・コサイン≈1 → novelty≈0 → a0≈0
        mem.save("novelty eighth", kind="observation")
        assert _a0("novelty eighth") < 0.2
    finally:
        for p in ps:
            p.stop()


def test_novelty_excludes_self_model():
    ps = _fixed_embed()
    for p in ps:
        p.start()
    try:
        mem = ObservationMemory()
        # self_model だけを溜める（母集合から除外される想定）
        for i in range(8):
            mem.save(f"novelty selfmodel {i}", kind="self_model")
        # 非 self_model の近傍は0件 → 除外が効けば既定0.5 → a0=0.75
        mem.save("novelty against selfmodel", kind="observation")
        assert _a0("novelty against selfmodel") == pytest.approx(0.75, abs=0.05)
    finally:
        for p in ps:
            p.stop()


def test_content_novelty_facade_empty_returns_default():
    from familiar_agent.config import MemoryConfig
    ps = _fixed_embed()
    for p in ps:
        p.start()
    try:
        mem = ObservationMemory()
        assert mem.content_novelty("") == pytest.approx(MemoryConfig().novelty_default)
        assert mem.content_novelty("   ") == pytest.approx(MemoryConfig().novelty_default)
    finally:
        for p in ps:
            p.stop()


@pytest.mark.asyncio
async def test_turn_arousal_prefers_user_input_else_final_text():
    from unittest.mock import AsyncMock

    from familiar_agent.agent import EmbodiedAgent

    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    agent._memory = MagicMock()
    agent._memory.content_novelty_async = AsyncMock(
        side_effect=lambda c: 0.9 if c == "user says X" else 0.1
    )
    # user_input があればそれで測る
    assert await agent._turn_arousal("user says X", "agent text") == pytest.approx(0.9)
    # 自発ターン（user_input 空）は final_text へフォールバック
    assert await agent._turn_arousal("   ", "agent text") == pytest.approx(0.1)


def test_config_novelty_defaults(monkeypatch):
    for k in ("NOVELTY_K", "NOVELTY_W_N", "NOVELTY_DEFAULT", "NOVELTY_A0_CAP"):
        monkeypatch.delenv(k, raising=False)
    from familiar_agent.config import MemoryConfig

    cfg = MemoryConfig()
    assert cfg.novelty_k == 7
    assert cfg.novelty_w_n == pytest.approx(1.5)
    assert cfg.novelty_default == pytest.approx(0.5)
    assert cfg.novelty_a0_cap == pytest.approx(1.5)
