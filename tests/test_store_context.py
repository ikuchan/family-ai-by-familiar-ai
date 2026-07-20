"""Tests for store/context.py（層の共有文脈）と、層の脱 mixin 化（C1）.

mixin は宿主の名前空間を共有するので、層どうしが名前で衝突しうる。実際 S6b で、
キュー層に置いた宣言が観測層の本体を MRO で覆い隠し、実体化が
NotImplementedError になる状態を作った（不変条件で気づいた）。

層を普通のクラスにし、**使うものを引数で受け取る**形へ変える。依存が構造として
見えるので、同じ事故が起きなくなる。共有するのは接続・ロック・person・埋め込み器の
4つだけで、それを `StoreContext` に束ねる。
"""

from __future__ import annotations

import pathlib
import re

from familiar_agent.store.context import StoreContext


def test_context_carries_only_what_layers_share() -> None:
    """共有するものが4つに絞られている（増やすときは意図的に）。"""
    fields = set(StoreContext.__dataclass_fields__)
    assert fields == {"db", "lock", "person_id", "embedder"}, fields


def test_context_provides_a_connection() -> None:
    """層は接続の作り方を知らず、文脈から受け取る。"""
    assert callable(getattr(StoreContext, "conn", None))


def test_context_can_be_rebound_to_another_person() -> None:
    """person だけ差し替えた文脈を作れる（for_person の受け皿）。"""
    assert callable(getattr(StoreContext, "for_person", None))


def test_layers_are_no_longer_mixins() -> None:
    """store と legacy に Mixin が残っていない（積み上げをやめた印）。"""
    for path in (
        "src/familiar_agent/store/jobs.py",
        "src/familiar_agent/store/situated.py",
        "src/familiar_agent/store/observations.py",
        "src/familiar_agent/legacy/semantic_layer.py",
    ):
        src = pathlib.Path(path).read_text()
        assert not re.search(r"class \w*Mixin\b", src), f"{path} に Mixin が残っている"


def test_layers_declare_their_dependencies_in_the_constructor() -> None:
    """層をまたぐ依存が引数に出ている（宿主の名前空間を覗かない）。"""
    from familiar_agent.legacy.semantic_layer import LegacySemanticLayer
    from familiar_agent.store.jobs import JobQueue
    from familiar_agent.store.observations import ObservationStore
    from familiar_agent.store.situated import SituatedVectors

    import inspect

    assert "observations" in inspect.signature(JobQueue.__init__).parameters
    obs_params = inspect.signature(ObservationStore.__init__).parameters
    assert "situated" in obs_params and "legacy" in obs_params
    for cls in (SituatedVectors, LegacySemanticLayer):
        assert "ctx" in inspect.signature(cls.__init__).parameters


def test_no_layer_borrows_from_a_host() -> None:
    """「宿主が実装する」宣言が消えている（借り物が無くなった）。"""
    for path in (
        "src/familiar_agent/store/jobs.py",
        "src/familiar_agent/store/situated.py",
        "src/familiar_agent/store/observations.py",
        "src/familiar_agent/legacy/semantic_layer.py",
    ):
        src = pathlib.Path(path).read_text()
        assert "宿主が実装する" not in src, f"{path} に借り物の宣言が残っている"
