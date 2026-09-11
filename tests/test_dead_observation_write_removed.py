"""到達しない `観察` の書き込みを撤去した。

`_run_post_response_pipeline` は `camera_used` が真のときだけ `direction="観察"` を
書いていた。ところが呼び出し元は `loop/event_loop.py` の1箇所だけで、そこは
`camera_used=False` を渡す。**この書き込みは一度も到達しない。**

書かれる中身も `final_text`（自分の応答テキスト）で、同じテキストは
`direction="発話"` の「自分が答えた：…」として既に書かれている。

見た印は `InformationProcessing._write_seen_mark` が書く（`test_seen_mark_record` に
理由がある）。旧 `run()` 時代の `観察` と同じ意味の記録で、定点名が加わる。
"""

from __future__ import annotations

import inspect


def test_the_pipeline_does_not_write_an_observation() -> None:
    """ターン後の処理が `観察` を**書かない**。

    **語ではなく構文で見る。** 環-e-い で、同じ処理が新しさを測るために `観察` を
    **読む**ようになった（`Cue(direction="観察")`）。語で探すとその読みに当たる。
    """
    import ast
    import textwrap

    from familiar_agent.agent import EmbodiedAgent

    tree = ast.parse(textwrap.dedent(inspect.getsource(EmbodiedAgent._run_post_response_pipeline)))
    writes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", "")
        if name not in ("write", "save", "save_async", "save_async_with_id"):
            continue
        if "観察" in ast.unparse(node):
            writes.append(ast.unparse(node)[:60])
    assert not writes, f"到達しない観察の書き込みが残っている: {writes}"


def test_the_loop_writes_the_seen_mark_instead() -> None:
    """見た印はループ側が書く。"""
    from familiar_agent.loop.event_loop import InformationProcessing

    assert hasattr(InformationProcessing, "_write_seen_mark"), "見た印の書き手が無い"
    src = inspect.getsource(InformationProcessing._write_seen_mark)
    assert 'direction="観察"' in src, "観察として書いていない"
