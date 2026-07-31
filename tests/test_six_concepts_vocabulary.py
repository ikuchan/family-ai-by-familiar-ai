"""6概念の呼び名が混ざっていないことの見張り。

`activation`・`a`・`score` という語に、実装上5つ以上の別の量が相乗りしていた。同じ語が
別の中身を指していると、読む側が取り違える。日本語・英語・記号の頭文字をすべて分けた。

| 概念 | 日本語 | 英語 | 記号 |
|---|---|---|---|
| 取込値と参照回数から導く量（時間で減らない） | 根づき | `groundedness` | `g` |
| 感情 PAD の A 軸 | 高ぶり | `arousal` | `a` |
| 競合候補の基礎強度 | 勢い | `dynamism` | `d` |
| 時間経過を含んだ重要度（関連を含まない） | 地力 | `merit` | `m` |
| W を DB へ溜めた旧方式 | 顕著性 | `salience` | `s` |
| W を選ぶ最終値（関連を含む） | 適合度 | `fit` | `f` |

このテストは、撤去した呼び名がコードへ戻っていないことだけを見る。個々の改名が正しく
効いているかは、概念ごとのテスト（`test_groundedness_rename` など）が担う。
"""

from __future__ import annotations

import pathlib
import re

import pytest

# 旧い呼び名と、それが指していた概念。除外は理由を1件ずつ挙げる。
_RETIRED = {
    "覚醒": "高ぶり",
    "喚起": "高ぶり",
    "activation_a0": "groundedness_g0",
    "activation_n": "groundedness_n",
    "_derive_activation": "_derive_groundedness",
    "memory_activation": "memory_salience",
}

# 旧名を**検証の対象として**文字列で持つファイル。
_ALLOWED = {
    "test_six_concepts_vocabulary.py",          # このテスト自身
    "test_groundedness_rename.py",              # 根づきの改名を旧名で確かめる
    "test_salience_rename.py",                  # 顕著性の改名を旧名で確かめる
    "test_dynamism_rename.py",                  # 勢いの改名を旧名で確かめる
    "test_fit_and_merit_rename.py",             # 適合度の改名を旧名で確かめる
    "test_self_state_removed.py",               # 撤去した自己状態の軸名を持つ
    "test_migration_029_utc_text_timestamps.py",  # 凍結マイグレーションの前提を確かめる
}


def _sources() -> list[pathlib.Path]:
    root = pathlib.Path(__file__).resolve().parents[1]
    paths = list((root / "src").rglob("*.py")) + list((root / "tests").rglob("*.py"))
    # マイグレーションは過去の一度きりの実行を再現する凍結物なので対象外。
    return [p for p in paths if p.name not in _ALLOWED]


@pytest.mark.parametrize("retired,replacement", sorted(_RETIRED.items()))
def test_retired_name_is_gone(retired, replacement) -> None:
    """撤去した呼び名がコードへ戻っていない。"""
    root = pathlib.Path(__file__).resolve().parents[1]
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(retired)}(?![A-Za-z0-9_])")
    stale = [str(p.relative_to(root)) for p in _sources()
             if pattern.search(p.read_text(encoding="utf-8"))]
    assert not stale, (
        f"旧名『{retired}』が残っている（→『{replacement}』）:\n" + "\n".join(sorted(stale))
    )


def test_the_six_names_all_resolve() -> None:
    """6概念の名前が実際に引ける（表と実装がずれていない）。"""
    from familiar_agent.config import MemoryConfig, RecallWeights
    from familiar_agent.tools.memory import _derive_groundedness, _score_breakdown
    from familiar_agent.workspace import Coalition

    assert _derive_groundedness(0.5, 0) == pytest.approx(0.5)          # 根づき
    assert MemoryConfig().recall_w_g > 0                               # 根づきの重み
    assert "w_g" in RecallWeights.__dataclass_fields__
    assert "dynamism" in Coalition.__dataclass_fields__                # 勢い
    parts = _score_breakdown(0.5, None, None, 0, 1.0, 0,
                             half_life_days=3.0, floor=0.001)
    assert hasattr(parts, "m") and hasattr(parts, "fit")               # 地力・適合度
    assert hasattr(parts, "g")                                         # 根づきの軸
