"""#11 段階5：イベント駆動ループを既定にし、旧 run() は `EVENT_LOOP=0` で残す。

撤去ではなく既定の反転。旧経路は比較対象として要る（`/speaker` の配線・会話履歴・
自己認識の注入は、いずれも旧経路と突き合わせて欠落を見つけた）。削除は #12。

アイドル側が持っていた自発系（drive の蓄積と発火・deferred の配信・旧 DesireSystem の
ターン）は #12a で撤去した。同じ役目は T（Tonic）と完了キューが持つ。
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
