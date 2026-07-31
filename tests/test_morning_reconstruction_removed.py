"""朝の再構成と、旧 ReAct のプロンプト断片の撤去。

`_morning_reconstruction`（朝の再構成）は、起動時に記憶を6経路から読み直して文脈を組む
ものだったが、イベント駆動ループへ移ってから**生きた呼び出し元が無い**（テストからしか
呼ばれていなかった）。同じことが要るなら MI の想起で実現する方針である。

一緒に消えるのは、`_morning_reconstruction` からしか呼ばれていなかった
`_select_context_blocks`（文脈ブロックの予算選択）と `_backfill_day_summaries`
（日次要約の穴埋め）。

旧 ReAct のプロンプト断片3つ（`_get_body_description`／`_exploration_context`／
`_self_continuity_context`）は、`_system_prompt` の撤去で呼び出し元を失っていた。定義
だけが残っていた。

`_generate_day_summary` は**残す**。`close()`（終了時）からも呼ばれており、そちらは
生きている。
"""

from __future__ import annotations

import pytest

from familiar_agent.agent import EmbodiedAgent

_REMOVED = [
    "_morning_reconstruction",
    "_select_context_blocks",
    "_backfill_day_summaries",
    "_get_body_description",
    "_exploration_context",
    "_self_continuity_context",
]


@pytest.mark.parametrize("name", _REMOVED)
def test_removed(name) -> None:
    """撤去したメソッドは残っていない。"""
    assert not hasattr(EmbodiedAgent, name), f"{name} が残っている"


def test_day_summary_generation_survives() -> None:
    """日次要約の生成は残す（`close()` から呼ばれる生きた経路がある）。"""
    assert hasattr(EmbodiedAgent, "_generate_day_summary"), (
        "_generate_day_summary まで消えている"
    )


def test_close_still_generates_the_day_summary() -> None:
    """終了時に日次要約を作る経路が繋がったままであること。"""
    import inspect

    src = inspect.getsource(EmbodiedAgent.close)
    assert "_generate_day_summary" in src, "close() から日次要約の生成が消えている"
