"""在席の判定（社会的発話・deferred 配信の共通ゲート）。

カメラが有効なときは顔検出だけを見て確定し、`_last_human_at` の分岐へ到達しなかった。
そのため、目の前で人が話しかけていても「誰も居ない」と判定され、返事まで保留になった
（実機で観測）。顔が見えることと、話しかけられていることは、どちらも在席の証拠である。
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from familiar_agent.agent import EmbodiedAgent


def _agent(*, watcher: bool, present: list[str], last_human: float | None) -> MagicMock:
    a = MagicMock()
    a._presence_watcher = MagicMock() if watcher else None
    a._pmm = MagicMock()
    a._pmm.get_present_ids = MagicMock(return_value=present)
    if last_human is None:
        del a._last_human_at
    else:
        a._last_human_at = last_human
    return a


def _permission(a) -> float:
    return EmbodiedAgent._social_presence_permission(a)


def test_recent_utterance_counts_as_presence_even_with_a_camera():
    # カメラが動いていて顔が見えていなくても、話しかけられていれば人は居る。
    a = _agent(watcher=True, present=[], last_human=time.time())
    assert _permission(a) == 1.0


def test_face_detection_counts_as_presence_without_any_utterance():
    a = _agent(watcher=True, present=["p1"], last_human=None)
    assert _permission(a) == 1.0


def test_empty_room_is_not_present():
    a = _agent(watcher=True, present=[], last_human=time.time() - 600)
    assert _permission(a) == 0.0


def test_no_camera_still_uses_the_utterance():
    a = _agent(watcher=False, present=[], last_human=time.time())
    assert _permission(a) == 1.0
