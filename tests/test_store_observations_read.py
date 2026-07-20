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

from familiar_agent.store.observations import ObservationStore
from familiar_agent.tools.memory import ObservationMemory


def test_observation_memory_holds_the_read_layer() -> None:
    assert hasattr(ObservationStore, "_read_observations_by_kind")
    assert hasattr(ObservationMemory, "get_dates_with_observations")


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


def test_reading_is_owned_by_the_store_layer() -> None:
    """読み出しの実装が層にあり、`memory.py` に二重に無い。

    層の内部ヘルパー（`_read_*`）は `memory.py` へ委譲しない（層の内側が外から
    触れる状態を残さないため）。よってここでは「memory.py に定義が無いこと」と
    「層に在ること」の両方を見る。
    """
    from familiar_agent.store.observations import ObservationStore

    src = pathlib.Path("src/familiar_agent/tools/memory.py").read_text()
    for name in ("_read_observations_by_kind", "_read_observations_by_situated",
                 "_read_supersede_chain"):
        assert hasattr(ObservationStore, name), f"{name} が層に無い"
        assert not re.search(rf"^    (?:async )?def {name}\b", src, re.M), \
            f"{name} が memory.py にも定義されている（二重実装）"

    # 日付系は外から呼ばれるので委譲が残る。ただし SQL は持たない。
    for name in ("get_dates_with_observations", "recall_on_this_day"):
        m = re.search(rf"^    (?:async )?def {name}\b.*?(?=^    (?:async )?def |\Z)", src, re.M | re.S)
        assert m, f"{name} の委譲が要る（外から呼ばれる）"
        assert "cur.execute" not in m.group(0), f"{name} の SQL が memory.py に残っている"
