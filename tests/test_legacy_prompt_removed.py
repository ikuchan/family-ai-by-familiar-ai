"""旧 ReAct のプロンプト組み立ての撤去。

`agent.run()` は特殊コマンドを処理したあと `InformationProcessing.run_iteration` へ委譲
するだけで、旧プロンプトの組み立てには到達しない。実行中のプロンプトは
`build_event_system_prompt`（`loop/prompt.py`）が組む。

撤去したのは次の3つで、いずれも生きた呼び出し元が無かった。

- `_system_prompt`：旧 ReAct のシステムプロンプト（安定部・可変部の対）を組む
- `_interoception`：内受容の felt-sense テキスト。呼び出し元は `_system_prompt` だけ
- `SYSTEM_PROMPT` と `MAX_ITERATIONS`：どちらも `_system_prompt` の中だけで使われていた

内受容そのものが無くなったわけではない。実行中の経路は `[内部状態(PI)]`（気分と欲求から
`event_loop._pi_ctx` が組む）が担う。
"""

from __future__ import annotations

import pathlib

import pytest


def test_legacy_prompt_builder_is_gone() -> None:
    """`_system_prompt` は無い。"""
    from familiar_agent.agent import EmbodiedAgent

    assert not hasattr(EmbodiedAgent, "_system_prompt"), "_system_prompt が残っている"


def test_interoception_is_gone() -> None:
    """`_interoception` は無い（`agent` からも `core.helpers` からも）。"""
    import familiar_agent.agent as agent_mod
    import familiar_agent.core.helpers as helpers_mod

    assert not hasattr(agent_mod, "_interoception"), "agent に _interoception が残っている"
    assert not hasattr(helpers_mod, "_interoception"), "helpers に _interoception が残っている"


def test_legacy_prompt_constants_are_gone() -> None:
    """`_system_prompt` の中だけで使われていた定数も消えている。"""
    import familiar_agent.agent as agent_mod

    assert not hasattr(agent_mod, "SYSTEM_PROMPT"), "SYSTEM_PROMPT が残っている"
    assert not hasattr(agent_mod, "MAX_ITERATIONS"), "MAX_ITERATIONS が残っている"


def test_event_prompt_is_the_live_path() -> None:
    """実行中のプロンプトは `build_event_system_prompt` が組む（撤去の前提）。"""
    from familiar_agent.loop.prompt import build_event_system_prompt

    stable, variable = build_event_system_prompt(
        self_understanding="me", family_md="family",
        present_ctx="present", pi_ctx="pi", workspace_ctx="w",
    )
    assert stable and variable, "実行中のプロンプトが組めていない"


def test_no_dangling_references() -> None:
    """撤去した名前がソースに残っていない。"""
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    stale = []
    for path in (root / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in ("_interoception", "_system_prompt"):
            # 語境界で見る。素朴な部分一致だと `build_event_system_prompt`（実行中の
            # プロンプトを組む生きた関数）が `_system_prompt` に当たってしまう。
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text):
                stale.append(f"{path.relative_to(root)}: {name}")
    assert not stale, "撤去した名前が残っている:\n" + "\n".join(sorted(set(stale)))


@pytest.mark.parametrize("supplier", ["_decayed_mood"])
def test_shared_suppliers_survive(supplier) -> None:
    """他からも使われている供給元は残す（芋づるで消さない）。"""
    from familiar_agent.agent import EmbodiedAgent

    assert hasattr(EmbodiedAgent, supplier), f"{supplier} まで消えている"
