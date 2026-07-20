"""Tests for store/observations.py の読み出し層（S6a）.

課題8 v0.6 で「層は生 SQL や WHERE を受けず、取り出しパターンごとの専用メソッドで
構造化値を受ける」と決めた。Phase 1 で `by_kind` 相当を4本だけ寄せ、残りは Phase 2
へ申し送っていた。ここでその宿題を果たす。

読み出しのパターンは3つに整理する。

- `by_kind`      種別と person で新しい順に読む
- `by_situated`  situated 相関で紐づく観測を読む（所有者に依らない母集合）
- `by_date`      日付・周期で読む（日ごとの一覧、その日の観測、記念日）

層は採点も想起判断も持たない（[D-データモデル]）。挙動は変えない。
"""

from __future__ import annotations

import pathlib
import re

from familiar_agent.store.observations import ObservationReadMixin
from familiar_agent.tools.memory import ObservationMemory


def test_observation_memory_inherits_the_read_layer() -> None:
    assert issubclass(ObservationMemory, ObservationReadMixin)


def test_public_entry_points_are_still_reachable() -> None:
    """agent.py が呼ぶ入口が残っている。"""
    for name in (
        "get_dates_with_observations",
        "get_dates_with_summaries",
        "get_observations_for_date",
        "delete_day_summaries_for_date",
        "recall_on_this_day_async",
        "get_earliest_date_async",
    ):
        assert hasattr(ObservationMemory, name), name


def test_read_layer_has_no_scoring() -> None:
    """層は採点を持たない（想起の産物は層の外で付ける）。

    判定はコード部分に限る。docstring で「採点を持ち込まない」と説明している語に
    反応しないよう、モジュールの docstring を除いて見る。
    """
    import ast

    src = pathlib.Path("src/familiar_agent/store/observations.py").read_text()
    tree = ast.parse(src)
    if ast.get_docstring(tree):
        tree.body = tree.body[1:]
    code = ast.unparse(tree)
    for banned in ("_compute_final_score", "_score_breakdown", "_emotion_match", "min_score"):
        assert banned not in code, f"層に採点が混ざっている: {banned}"


def test_moved_readers_are_gone_from_memory_module() -> None:
    """移動元に定義が残っていない（二重定義の反証）。"""
    src = pathlib.Path("src/familiar_agent/tools/memory.py").read_text()
    for name in (
        "_read_observations_by_kind",
        "_read_observations_by_situated",
        "_read_supersede_chain",
        "get_dates_with_observations",
        "recall_on_this_day",
    ):
        assert not re.search(rf"^    def {name}\b", src, re.M), f"{name} の定義が残っている"
