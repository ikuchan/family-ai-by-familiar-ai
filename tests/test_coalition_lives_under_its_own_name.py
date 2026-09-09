"""候補の器に、実態どおりの名前を与える（環-e-に の後始末）。

`familiar_agent/workspace.py` は、競合と放送を行っていた `GlobalWorkspace` を #12a で
撤去したあとの残りで、**中身は候補の器（`Coalition`）だけ**である。W を組み立てる役目は
持っていない。

に-5-に-2 で W を組む `loop/workspace.py` を作ったので、**`workspace.py` が2つ**になった。
package が違うので import は衝突しないが、開いた人は同じ名前の file を2つ見ることになる。
**名前が実態と合っていないほうを直す。**

これは環-g（ループの語を束ねる）と同じ性質の作業である——同じものに複数の名前が付く／
違うものに同じ名前が付く、のどちらも読み手の負担になる。**挙動は変えない。**
"""

from __future__ import annotations

import importlib


def test_the_coalition_lives_in_its_own_module():
    mod = importlib.import_module("familiar_agent.coalition")
    assert hasattr(mod, "Coalition")


def test_the_old_name_is_gone():
    """**旧名で引けないことを確かめる。** 数え上げでは網羅の証明にならない。"""
    try:
        importlib.import_module("familiar_agent.workspace")
    except ModuleNotFoundError:
        return
    raise AssertionError("旧名 `familiar_agent.workspace` がまだ引ける")


def test_the_workspace_name_now_means_one_thing():
    """`workspace` という名前が指すのは、W を組む側だけになる。"""
    from familiar_agent.loop import workspace

    assert hasattr(workspace, "compose"), "W を組む側が `workspace` である"
