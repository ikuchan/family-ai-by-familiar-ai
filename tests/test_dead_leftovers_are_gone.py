"""読み手の居ないものを落とす（環-e-に の後始末）。

コメントの点検で、**コメントではなくコードのほうが死んでいた**ものが2つ出た。どちらも
撤去した機構のなごりで、残しておくと「まだ効いている」と読まれる。

| 落とすもの | なぜ死んでいるか |
|---|---|
| `coalition.py` の `_MIN_THRESHOLD`・`_ERROR_SENSITIVITY` | `GlobalWorkspace`（#12a で撤去）の発火閾値。参照 0 件 |
| `_run_post_response_pipeline` の `close_parent_id` | 引数として受け取るが、**177 行の本体で一度も読んでいない**。呼び手は毎回渡していた |

`close_parent_id` は「親が決着したら生きた子をまとめて閉じる」という、**版チェーンに
置き換わって消えた仕組み**の名残である（`_write_version` の docstring が「親子をまとめて
畳む操作は要らない（撤去済み）」と書いている）。渡しているのに読まれない引数は、
**読む人に「閉じている」と誤解させる**。
"""

from __future__ import annotations

import inspect


def test_the_ignition_thresholds_are_gone():
    """`GlobalWorkspace` の発火閾値。撤去した機構の設定値だけが残っていた。"""
    from familiar_agent import coalition

    src = inspect.getsource(coalition)
    assert "_MIN_THRESHOLD" not in src
    assert "_ERROR_SENSITIVITY" not in src


def test_the_container_still_scores():
    """**反証側**：器そのものは生きている（6モジュールが使う）。"""
    from familiar_agent.coalition import Coalition

    c = Coalition(source="s", summary="x", dynamism=1.0, urgency=0.0, novelty=0.0, context_block="")
    assert c.score() > 0


def test_the_pipeline_no_longer_takes_a_parent_to_close():
    from familiar_agent.agent import EmbodiedAgent

    params = inspect.signature(EmbodiedAgent._run_post_response_pipeline).parameters
    assert "close_parent_id" not in params


def test_the_loop_no_longer_passes_it():
    from familiar_agent.loop.event_loop import InformationProcessing

    assert "close_parent_id" not in inspect.getsource(InformationProcessing._finish)
