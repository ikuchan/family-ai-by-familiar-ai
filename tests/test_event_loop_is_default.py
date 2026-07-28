"""#11 段階5：イベント駆動ループを既定にし、旧 run() は `EVENT_LOOP=0` で残す。

撤去ではなく既定の反転。旧経路は比較対象として要る（`/speaker` の配線・会話履歴・
自己認識の注入は、いずれも旧経路と突き合わせて欠落を見つけた）。削除は #12。

既定を反転すると、GUI のアイドルループとイベント駆動ループが**同じ役目を二重に持つ**。
drive の蓄積と発火（T が回す）・deferred の配信（QC へ届く）がそれで、アイドル側は
入口で1回判定してまとめて飛ばす。
"""

from __future__ import annotations

import os
from unittest.mock import patch

from familiar_agent.config import AgentConfig


def test_event_loop_is_on_by_default():
    with patch.dict(os.environ, {}, clear=True):
        assert AgentConfig().event_loop is True


def test_legacy_path_is_still_reachable_by_env():
    with patch.dict(os.environ, {"EVENT_LOOP": "0"}, clear=True):
        assert AgentConfig().event_loop is False


def test_idle_loop_skips_self_initiated_work_when_the_new_loop_is_on():
    import inspect

    from familiar_agent.gui import FamiliarWindow

    src = inspect.getsource(FamiliarWindow._process_queue)
    # 判定は1箇所だけ（#12 でこの塊ごと切り取れるように）。
    assert src.count('"event_loop", False') == 1
    # drive の発火と deferred の配信はゲートより後＝新ループでは走らない。
    assert src.index('"event_loop", False') < src.index("should_deliver_deferred_result")
