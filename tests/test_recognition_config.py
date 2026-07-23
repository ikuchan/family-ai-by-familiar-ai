"""話者自動確定のしきい値と在席巡回間隔の Config 既定（誤確定・高頻度巡回の是正）。

- face_switch_threshold：顔認識で話者を自動確定するほど確信あるかの床。誤確定を減らす。
- presence_interval_sec：CameraPresenceWatcher の巡回周期（Config 化・高頻度を是正）。
"""

from __future__ import annotations

import os
from unittest.mock import patch

from familiar_agent.config import RecognitionConfig


def test_face_switch_threshold_default_is_065():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("FACE_SWITCH_THRESHOLD", None)
        assert RecognitionConfig().face_switch_threshold == 0.65


def test_presence_interval_default_is_30():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PRESENCE_INTERVAL_SEC", None)
        assert RecognitionConfig().presence_interval_sec == 30.0


def test_presence_interval_reads_env():
    with patch.dict(os.environ, {"PRESENCE_INTERVAL_SEC": "45"}, clear=False):
        assert RecognitionConfig().presence_interval_sec == 45.0
