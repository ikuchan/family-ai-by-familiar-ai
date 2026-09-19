"""環-p-ろ（パネルにセンサの読みと話者の根拠）・環-q の監視（音量）・アラームの名前の検め（2026-09-19）。

- パネルは PMM の在席表しか映さず、センサが人を見ていても「（在席者なし）」に見えた（実機 11:34）。
  センサの最新の読み（定点・人数・何秒前）と話者の根拠（/speaker から N 秒・返事から N 秒・在席表・切れている）を出す。
- YVC-300 のミキサーが 40% で音がほぼ無音だった（環-q）。起動時に読んで 50% 未満なら知らせる。値は変えない。
- アラームの `label` もタイマー・ストップウォッチと同じ検め。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.core.audio_volume import parse_amixer
from familiar_agent.gui import format_sensor_rows

AMIXER = """Simple mixer control 'PCM',0
  Capabilities: pvolume pvolume-joined pswitch pswitch-joined
  Playback channels: Mono
  Limits: Playback 0 - 50
  Mono: Playback 20 [40%] [-30.00dB] [on]
"""


# ── 音量 ────────────────────────────────────────────────────────────────


def test_parse_amixer_reads_the_percent():
    assert parse_amixer(AMIXER) == 40
    assert (
        parse_amixer("Simple mixer control 'PCM',0\n  Mono: Playback 50 [100%] [0.00dB] [on]\n")
        == 100
    )
    assert parse_amixer("") is None


def test_the_note_warns_below_half_and_is_quiet_otherwise(monkeypatch):
    from familiar_agent.agent import EmbodiedAgent

    a = EmbodiedAgent.__new__(EmbodiedAgent)
    monkeypatch.setattr("familiar_agent.core.audio_volume.read_percent", lambda name: 40)
    note = a.audio_volume_note()
    assert "40%" in note and "YVC-300" in note and "amixer" in note
    monkeypatch.setattr("familiar_agent.core.audio_volume.read_percent", lambda name: 80)
    assert a.audio_volume_note() == ""
    monkeypatch.setattr("familiar_agent.core.audio_volume.read_percent", lambda name: None)
    assert a.audio_volume_note() == ""  # 機器が無ければ黙る


# ── パネル ──────────────────────────────────────────────────────────────


def test_sensor_rows_name_the_pose_count_age_and_speaker_basis():
    rows = format_sensor_rows(("押入", 2, 12.4), "/speaker から 45 秒")
    assert rows == ["センサ：押入 に 2 人（12 秒前）", "話者の根拠：/speaker から 45 秒"]
    assert format_sensor_rows(("正面", 0, 3.0), "切れている") == [
        "センサ：正面 に誰も居ない（3 秒前）",
        "話者の根拠：切れている",
    ]
    assert format_sensor_rows(None, "") == ["センサ：（まだ見ていない）", "話者の根拠：—"]


def test_the_sensor_remembers_its_last_reading(monkeypatch):
    from familiar_agent.poses import Pose
    from familiar_agent.presence_sensor import PresenceSensor

    camera = MagicMock()
    camera.position = AsyncMock(return_value=(0.0, -0.5))
    camera.capture = AsyncMock(return_value=("B64", "/tmp/f.jpg"))
    detector = MagicMock()
    detector.boxes = AsyncMock(return_value=[(0.0, 0.0, 10.0, 10.0), (50.0, 0.0, 60.0, 10.0)])
    s = PresenceSensor(
        camera=camera,
        poses_getter=AsyncMock(return_value=[Pose("正面", 0.0, -0.5)]),
        detector=detector,
        tolerance=0.02,
        window_sec=120.0,
        interval_sec=30.0,
    )
    assert s.last_reading() is None
    clock = {"t": 1000.0}
    monkeypatch.setattr("familiar_agent.presence_sensor.time.time", lambda: clock["t"])
    asyncio.run(s.check_once())
    clock["t"] = 1012.0
    assert s.last_reading() == ("正面", 2, 12.0)


def test_speaker_basis_follows_speaker_known():
    from familiar_agent.agent import EmbodiedAgent

    a = EmbodiedAgent.__new__(EmbodiedAgent)
    a._pmm = MagicMock()
    a._pmm.current_speaker_id = "p1"
    a._pmm.get_present_ids = MagicMock(return_value=[])
    a.config = MagicMock()
    a.config.presence_said_sec = 60.0
    now = time.time()
    a._speaker_set_at = now - 45.0
    a._speaker_confirmed_at = None
    assert a.speaker_basis().startswith("/speaker から 45 秒")
    a._speaker_set_at = now - 200.0
    a._speaker_confirmed_at = now - 12.0
    assert a.speaker_basis().startswith("返事から 12 秒")
    a._speaker_confirmed_at = now - 200.0
    a._pmm.get_present_ids = MagicMock(return_value=["p1"])
    assert a.speaker_basis() == "在席表（顔照合か /speaker）"
    a._pmm.get_present_ids = MagicMock(return_value=[])
    assert a.speaker_basis() == "切れている"
    a._pmm.current_speaker_id = None
    assert a.speaker_basis() == "—"


# ── アラームの名前 ───────────────────────────────────────────────────────


def test_alarm_labels_are_checked_too(monkeypatch):
    from tests.test_alarm import _tool

    t, store, _, _ = _tool()
    asyncio.run(t.call("set_alarm", {"at": "21:00", "label": "何のために起こすか不明"}))
    assert store.active()[0]["label"] == "アラーム"
