"""在席をリアルタイムに近づける（知-ai 段 1・2026-10-05・本人の決定・`設計方針_在席と顔ぶれ` v0.1）。

在席（不特定の誰かがいるか）はカメラだけで決める。30 秒ごとに確かめ、最後に人を見てから 60 秒は居るとみなして
いたので、居なくなったことに気づくのが最大 1 分半遅れた。確かめる間隔を 3 秒、滞留窓を 30 秒にする（仮の値）。滞留窓は 30 秒から 10 秒にした（知-bg・2026-10-19・本人の決定）。
"""

from __future__ import annotations

from familiar_agent.config import CameraConfig


def test_occupancy_is_checked_every_three_seconds(monkeypatch):
    monkeypatch.delenv("CAMERA_OCCUPANCY_INTERVAL", raising=False)
    interval = CameraConfig().occupancy_interval_sec  # 設定を丸ごと表示させない
    assert interval == 3.0


def test_the_dwell_is_ten_seconds(monkeypatch):
    monkeypatch.delenv("CAMERA_OCCUPANCY_DWELL", raising=False)
    dwell = CameraConfig().occupancy_dwell_sec  # 設定を丸ごと表示させない
    assert dwell == 10.0
