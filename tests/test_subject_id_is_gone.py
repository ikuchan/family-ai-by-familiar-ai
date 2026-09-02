"""`subject_id` の受け渡しを撤去する（段5 の残り）。

**列は 056 で落とした。** 誰についての記録かは面（`about`）が持つ。だが引数だけが残り、
`materialize_save_event` は受け取って**捨てて**いた。呼び出し側は渡し続けていたので、
読むと「まだ意味がある値」に見える——次に触る者が誤る形である。

撤去の根拠は 042・056 で測ったとおり。`subject_id` が実在の人を指す観測は 397 件で、
**その全件がその人の面を既に持っていた**（`present` 337／`about` 79／`addressee` 35／
`actor` 26／`source` 9／`beneficiary` 2）。写す先が無い。
"""

from __future__ import annotations

import ast
import inspect
import pathlib


def _params_of(func) -> set[str]:
    tree = ast.parse(inspect.cleandoc(inspect.getsource(func)))
    fn = tree.body[0]
    assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
    args = fn.args
    return {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}


def test_the_store_no_longer_takes_a_subject() -> None:
    from familiar_agent.store.observations import ObservationStore

    assert "subject_id" not in _params_of(ObservationStore.materialize_save_event)


def test_the_memory_facade_no_longer_takes_a_subject() -> None:
    """`save_async_with_id` は `**kw` 渡しなので、明示の引数を持つ `save` を見る。"""
    from familiar_agent.tools.memory import ObservationMemory

    assert "subject_id" not in _params_of(ObservationMemory.save)


def test_nobody_passes_a_subject_any_more() -> None:
    """本番コードに `subject_id=` が1つも残っていない（説明文の言及は数えない）。"""
    hits = []
    for path in sorted(pathlib.Path("src").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "subject_id":
                hits.append(f"{path}:{node.value.lineno}")
    assert not hits, hits
