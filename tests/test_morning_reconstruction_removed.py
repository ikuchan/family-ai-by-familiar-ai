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

`_generate_day_summary`（終了時の日次要約）と `_write_today_narrative`（日記）も 2026-09-14 に
撤去した。日次の畳み込みは REST 内省の層 1（`loop/rest_fold.py`・記-a-ろ-は）が担い、
`self_narrative` は層 1 で代替する（環-d）。
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


def test_the_shutdown_no_longer_writes_a_day_summary_or_a_diary() -> None:
    """終了時の日次要約と日記は撤去（記-a-ろ-は・2026-09-14）。畳むのは REST 内省の仕事。"""
    import inspect

    for name in ("_generate_day_summary", "_write_today_narrative"):
        assert not hasattr(EmbodiedAgent, name), f"{name} が残っている"
    src = inspect.getsource(EmbodiedAgent.close)
    assert "day_summar" not in src and "narrative" not in src
