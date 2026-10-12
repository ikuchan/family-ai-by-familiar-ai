"""長い手がかりの想起を軽くする（出-bz-は・2026-10-13・本人の決定）。

調べものの結果が届いたあとの想起は、手がかり＝版の文（最大 8,192 字超）で引く。10/13 01:55 は頭で 1.33 秒、
03:55 は 3.47 秒止まった。本番の記録 4,748 件で測ると、語の軸が語ごとの当たりを数える問い合わせは語の数に比例し、
15 語で 0.12 秒、303 語で 1.67 秒だった。03:55 は版が 8,192 字で切られて記録され、手がかりは全文のままだったので、
版の埋め込みの控えも当たらなかった。

直すのは 2 つだけで、どちらも想起の中に閉じる（`req.cue`・Jev・主LLM へ渡す文は変えない）。
- 語の軸は、文に出てくる順で先頭から 20 語だけ使う（本人の決定：N=20）。版の文は人の問いから始まるので、問いの語が残る。
- 手がかりの埋め込みは、版を書くときと同じ上限で切った文で計算する。
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from familiar_agent.config import AgentConfig
from familiar_agent.core import keyword_rules
from familiar_agent.core.keyword_rules import pick_words
from familiar_agent.tools import memory as memory_mod
from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

_PLACES = (
    "東京 大阪 京都 札幌 仙台 名古屋 神戸 福岡 広島 横浜 千葉 埼玉 奈良 金沢 新潟 長野 静岡 岡山 熊本 鹿児島 "
    "那覇 函館 青森 盛岡 秋田 山形 福島 水戸 宇都宮 前橋"
).split()
_NOUNS_TEXT = "".join(f"{p}に行った。" for p in _PLACES)
_LIMIT = AgentConfig().completion_content_max


def _long_text() -> str:
    text = "「明日の天気は？」と聞かれ、" + _NOUNS_TEXT + "です。" * 3000
    assert len(text) > _LIMIT
    return text


# ── ① 語の軸は先頭から 20 語 ─────────────────────────────────────────


def test_words_are_capped_in_order():
    words = pick_words(_NOUNS_TEXT)
    assert keyword_rules.WORDS_MAX == 20
    assert words == _PLACES[:20]


def test_a_short_text_keeps_all_its_words():
    """**反証の側**：20 語に満たなければ、そのまま返る。"""
    assert pick_words("".join(f"{p}に行った。" for p in _PLACES[:15])) == _PLACES[:15]


# ── ② 埋め込みの手がかりだけを切る ─────────────────────────────────────


def _recall_seen(query: str):
    seen: dict = {"embedded": [], "worded": []}

    def encode_query(self, texts):
        seen["embedded"].extend(texts)
        return [[1.0, 0.0, 0.0]] * len(texts)

    def fake_pick(text):
        seen["worded"].append(text)
        return []

    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[1.0, 0.0, 0.0]]),
        patch.object(_EmbeddingModel, "encode_query", encode_query),
        patch.object(memory_mod, "pick_words", fake_pick),
    ):
        ObservationMemory().recall(query, n=5, min_score=0.0)
    return seen


def test_only_the_embedded_cue_is_cut():
    text = _long_text()
    seen = _recall_seen(text)
    assert seen["embedded"] == [text[:_LIMIT]]
    assert seen["worded"] == [text]  # 語の軸には全文（語の数は pick_words が絞る）


def test_a_short_cue_is_embedded_as_is():
    """**反証の側**：上限より短い手がかりは切らない。"""
    seen = _recall_seen("明日の天気は？")
    assert seen["embedded"] == ["明日の天気は？"]


# ── ③ 版の埋め込みの控えが当たる ─────────────────────────────────────


@pytest.fixture()
def counted_model():
    calls: list[list[str]] = []

    class _Model:
        def encode(self, texts, **kw):
            calls.append(list(texts))
            v = np.ones((len(texts), 1024), dtype=np.float32)
            return v / np.linalg.norm(v, axis=1, keepdims=True)

    return _Model(), calls


def test_the_version_vector_is_reused_for_a_long_cue(counted_model):
    model, calls = counted_model
    text = _long_text()
    with patch.object(_EmbeddingModel, "pre_warm"):
        mem = ObservationMemory()
    embedder = mem._embedder
    embedder._mr_model = model
    embedder.encode_document([text[:_LIMIT]])  # 版を書いたときの埋め込み
    with patch.object(memory_mod, "pick_words", lambda t: []):
        mem.recall(text, n=5, min_score=0.0)
    assert calls == [[text[:_LIMIT]]]  # 想起ではモデルを呼ばない
