"""Tests for emotion_pad（PAD↔ラベルの生きた正本と逆引き・W2a・未接続）。

`emotion_pad.LABEL_PAD` が12ラベル→4軸 PAD の生きた正本。マイグレーション025 の
`_LABEL_PAD` は凍結写しで、両者は値一致する。`label_from_pad` は PAD を最近傍
（ユークリッド）で12ラベルへ量子化する。W2a では未接続（呼び出しは W2b）。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from familiar_agent.emotion_pad import LABEL_PAD, label_from_pad
from familiar_agent.mood_register import MoodPAD


_EXPECTED_LABELS = {
    "happy", "excited", "curious", "moved", "surprised", "nostalgic",
    "relieved", "tender", "playful", "proud", "sad", "neutral",
}


def _load_backfill_migration():
    path = Path(__file__).parent.parent / "migration" / "2026-07-16-025_backfill_emotion_pad.py"
    spec = importlib.util.spec_from_file_location("backfill_emotion_pad_migration", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ── 1. 正本の網羅と範囲 ─────────────────────────────────────────────────────
def test_label_pad_covers_expected_labels() -> None:
    assert set(LABEL_PAD.keys()) == _EXPECTED_LABELS
    for label, pad in LABEL_PAD.items():
        assert len(pad) == 4, label
        assert all(0.0 <= x <= 1.0 for x in pad), label


# ── 2. 正本と凍結写し（マイグレーション025）の一致 ─────────────────────────
def test_label_pad_matches_frozen_migration_copy() -> None:
    mod = _load_backfill_migration()
    assert LABEL_PAD == mod._LABEL_PAD


# ── 3. 表の厳密値 → そのラベル ──────────────────────────────────────────────
def test_label_from_pad_exact_values() -> None:
    assert label_from_pad(MoodPAD(0.80, 0.15, 0.55, 0.60)) == "happy"
    assert label_from_pad(MoodPAD(0.20, 0.75, 0.25, 0.30)) == "sad"
    assert label_from_pad(MoodPAD(0.75, 0.15, 0.55, 0.90)) == "proud"


# ── 4. 近傍 → 最近傍ラベル ──────────────────────────────────────────────────
def test_label_from_pad_nearest() -> None:
    # happy=(0.80,0.15,0.55,0.60) をわずかに動かす
    assert label_from_pad(MoodPAD(0.82, 0.13, 0.57, 0.58)) == "happy"


# ── 5. 中立 → neutral ───────────────────────────────────────────────────────
def test_label_from_pad_neutral() -> None:
    assert label_from_pad(MoodPAD()) == "neutral"
