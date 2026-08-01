"""場面イベントから欲求への結線を撤去した（知-a S1 の続き）。

`_react_to_scene_events` は `SceneTracker.update` が返すイベントを受けて
`greet_companion`／`worry_companion` を押していた。その `update` の呼び出しを外したので
（理由は `test_scene_tracker_call_removed` にある）、イベントは二度と来ない。

在/不在は `PresenceSensor`（YOLO）が担う。`知覚在席` §3-2 は在/不在を G（T 側・連続）の
担当と定めており、VLM が拾ったラベルで在席欲求を動かす経路は設計に無い。
"""

from __future__ import annotations

import pathlib


def test_the_helper_is_gone() -> None:
    """関数そのものが無い。"""
    from familiar_agent import agent
    from familiar_agent.core import helpers

    assert not hasattr(helpers, "_react_to_scene_events"), "純関数側に残っている"
    assert not hasattr(agent, "_react_to_scene_events"), "再輸出が残っている"


def test_no_source_mentions_it() -> None:
    """旧名がソースに1件も残っていない。"""
    root = pathlib.Path(__file__).resolve().parents[1]
    stale = []
    for path in (root / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if "_react_to_scene_events" in line:
                stale.append(f"{path.relative_to(root)}:{i}")
    assert not stale, "旧名が残っている:\n" + "\n".join(stale)
