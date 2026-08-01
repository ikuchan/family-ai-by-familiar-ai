"""ターン後の `SceneTracker.update` 呼び出しを止める（知-a S1 の範囲を絞る）。

`extract_entities` を直すと、`see` の意味づけと同時に**ターン後の経路も生き返る**。その
経路は設計が想定する入力を受け取っていない。

`_run_post_response_pipeline` を呼ぶのは `loop/event_loop.py` の1箇所だけで、そこは
`camera_image=None`・`camera_used=False`・`observation_action_name=None` を渡す。つまり
`SceneTracker.update` に渡るのはカメラ画像ではなく、**エージェント自身の発話テキスト**
（`final_text[:500]`）である。そこから抜いたラベルが `scene_entities` に入り、`person` を
含めば `_react_to_scene_events` が `greet_companion` を +0.6 する。

実機の llava:7b は `person`・`child`・`adult` を返すので、これは実際に起きる。

在/不在は S3（知-b）の YOLO が担い、実機で通し確認済みである（`課題8 §7`）。設計
（`知覚在席` §3-2）も在/不在を G（T 側・連続）の担当と定め、VLM は「意味づけ」に限る。
発話テキスト由来のラベルで在席欲求を動かす経路を残す理由がない。

`SceneTracker` クラスそのものは残す。止めるのは呼び出し1箇所である。
"""

from __future__ import annotations

import inspect
import pathlib


def test_the_pipeline_does_not_call_scene_update() -> None:
    """ターン後の処理が `SceneTracker.update` を呼ばない。"""
    from familiar_agent.agent import EmbodiedAgent

    src = inspect.getsource(EmbodiedAgent._run_post_response_pipeline)
    assert "_scene.update(" not in src, "ターン後の処理がまだ場面を更新している"


def test_no_source_calls_scene_update() -> None:
    """呼び出しがソースのどこにも残っていない。"""
    root = pathlib.Path(__file__).resolve().parents[1]
    stale = []
    for path in (root / "src").rglob("*.py"):
        if path.name == "scene.py":
            continue                      # 定義の側は残す
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if "_scene.update(" in line:
                stale.append(f"{path.relative_to(root)}:{i}")
    assert not stale, "場面の更新が残っている:\n" + "\n".join(stale)


def test_the_scene_tracker_itself_survives() -> None:
    """器は残す（撤去したのは呼び出しであって、場面追跡の設計ではない）。"""
    from familiar_agent.scene import SceneTracker

    assert hasattr(SceneTracker, "update"), "SceneTracker.update ごと消えている"


def test_react_to_scene_events_is_no_longer_wired() -> None:
    """場面イベントから欲求への結線も外す（イベントが二度と来ないため）。"""
    from familiar_agent.agent import EmbodiedAgent

    src = inspect.getsource(EmbodiedAgent._run_post_response_pipeline)
    assert "_react_to_scene_events" not in src, "呼ばれない結線が残っている"
