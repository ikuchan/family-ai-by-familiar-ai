"""動体検知（ONVIF PullPoint）→ 知覚ターン起動（案B）の純ロジック。

- Config：MOTION_WATCH 既定 off・デバウンス60秒・pull 60秒・内声あり。
- is_motion_active：motion トピック＋アクティブ状態のときだけ True（未知トピックは無視）。
- Debouncer：一度発火したら debounce 秒は再発火しない（バーストを1回にまとめる）。
"""

from __future__ import annotations

import os
from unittest.mock import patch

from familiar_agent.config import RecognitionConfig
from familiar_agent.recognition.motion_watcher import Debouncer, is_motion_active


# ── Config ───────────────────────────────────────────────────────────────────

def test_motion_config_defaults():
    with patch.dict(os.environ, {}, clear=False):
        for k in ("MOTION_WATCH", "MOTION_DEBOUNCE_SEC", "MOTION_PULL_TIMEOUT_SEC"):
            os.environ.pop(k, None)
        c = RecognitionConfig()
    assert c.motion_watch is False
    assert c.motion_debounce_sec == 60.0
    assert c.motion_pull_timeout_sec == 60.0
    assert c.motion_inner_voice.strip()


# ── is_motion_active ─────────────────────────────────────────────────────────

def test_motion_active_true_for_motion_topic_with_true_item():
    topic = "tns1:RuleEngine/CellMotionDetector/Motion"
    assert is_motion_active(topic, {"IsMotion": "true"}) is True


def test_motion_active_false_when_state_false():
    topic = "tns1:RuleEngine/CellMotionDetector/Motion"
    assert is_motion_active(topic, {"IsMotion": "false"}) is False


def test_motion_active_false_for_non_motion_topic():
    assert is_motion_active("tns1:Device/HardwareFailure", {"State": "true"}) is False


def test_motion_active_true_for_motion_topic_without_items():
    # 一部カメラは motion トピックだけ送る（明示 item なし）→ 発生とみなす
    assert is_motion_active("tns1:VideoSource/MotionAlarm", {}) is True


# ── Debouncer ────────────────────────────────────────────────────────────────

def test_debouncer_fires_first_then_suppresses_within_window():
    d = Debouncer(window_sec=60.0)
    assert d.allow(now=100.0) is True     # 初回は通す
    assert d.allow(now=130.0) is False    # 60秒以内は抑制
    assert d.allow(now=159.9) is False
    assert d.allow(now=160.0) is True     # 60秒経過で再度通す
    assert d.allow(now=170.0) is False    # また抑制


# ── agent 配線：コールバックが保留フラグを立てる ────────────────────────────

def test_note_motion_sets_pending():
    from unittest.mock import MagicMock

    from familiar_agent.agent import EmbodiedAgent

    s = MagicMock()
    s._motion_pending = False
    EmbodiedAgent._note_motion(s)
    assert s._motion_pending is True
