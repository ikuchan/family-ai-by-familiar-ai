"""知-bv：機器の見張り（`設計方針_機器の見張り`）。

本物の機器には触らない。音の命令（`wpctl`、`pw-dump`）は偽の口に差し替える。全体テストはこの機体で回るので、
試験が本物の YAMAHA の使い方を変えないため。
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from familiar_agent.config import DeviceWatchConfig
from familiar_agent.core.device_watch import CheckResult, next_interval
from familiar_agent.io import device_watch as dw

_CFG = DeviceWatchConfig()


# ── 間隔の決まり（本人の決定 (3)、仮の決定 (5)）────────────────────────────────


def test_the_interval_starts_at_five_minutes_and_doubles_up_to_four_hours() -> None:
    assert next_interval(None, CheckResult(), cfg=_CFG) == 300.0
    assert next_interval(300.0, CheckResult(), cfg=_CFG) == 600.0
    assert next_interval(10_000.0, CheckResult(), cfg=_CFG) == 14_400.0


def test_the_interval_goes_back_to_five_minutes_after_a_repair_or_a_failure() -> None:
    assert next_interval(14_400.0, CheckResult(repaired=["スピーカー"]), cfg=_CFG) == 300.0
    assert next_interval(14_400.0, CheckResult(failed=["カメラ"]), cfg=_CFG) == 300.0


def test_the_defaults_are_the_decided_values() -> None:
    assert (_CFG.first_delay_sec, _CFG.base_sec, _CFG.max_sec, _CFG.stale_sec) == (
        60.0,
        300.0,
        14_400.0,
        60.0,
    )


# ── スピーカー（§3.1）──────────────────────────────────────────────────────


def _dump(profile: str, *, sink: bool) -> str:
    """pw-dump の形の偽物。YVC-300 の機器（49）と、`sink` なら出口（59）。"""
    objs: list[dict] = [
        {
            "id": 49,
            "type": "PipeWire:Interface:Device",
            "info": {
                "props": {"device.description": "Yamaha YVC-300"},
                "params": {
                    "EnumProfile": [{"index": 0, "name": "off"}, {"index": 1, "name": "pro-audio"}],
                    "Profile": [{"index": 0 if profile == "off" else 1, "name": profile}],
                },
            },
        },
        {
            "id": 54,
            "type": "PipeWire:Interface:Node",
            "info": {"props": {"media.class": "Audio/Sink", "node.description": "内部オーディオ"}},
        },
    ]
    if sink:
        objs.append(
            {
                "id": 59,
                "type": "PipeWire:Interface:Node",
                "info": {
                    "props": {"media.class": "Audio/Sink", "node.description": "Yamaha YVC-300 Pro"}
                },
            }
        )
    return json.dumps(objs)


class _FakeAudio:
    """`wpctl` と `pw-dump` の偽物。使い方を変えると、`appear_after` 回あとの pw-dump で出口が現れる。"""

    def __init__(self, profile: str, default: str, *, appear_after: int = 1) -> None:
        self.profile = profile
        self.default = default
        self.calls: list[list[str]] = []
        self._dumps_since_change: int | None = None if profile == "off" else 99
        self._appear_after = appear_after

    def __call__(self, cmd: list[str]) -> str:
        self.calls.append(cmd)
        if cmd[:2] == ["wpctl", "inspect"]:
            return f'  * node.description = "{self.default}"\n'
        if cmd[:2] == ["wpctl", "set-profile"]:
            self.profile = "pro-audio" if cmd[3] == "1" else "off"
            self._dumps_since_change = 0
            return ""
        if cmd[:2] == ["wpctl", "set-default"]:
            self.default = {"59": "Yamaha YVC-300 Pro", "54": "内部オーディオ"}[cmd[2]]
            return ""
        if cmd == ["pw-dump"]:
            sink = False
            if self._dumps_since_change is not None:
                self._dumps_since_change += 1
                sink = self._dumps_since_change > self._appear_after
            return _dump(self.profile, sink=sink and self.profile == "pro-audio")
        raise AssertionError(cmd)


def test_the_speaker_is_fine_when_the_default_output_is_the_yamaha() -> None:
    audio = _FakeAudio("pro-audio", "Yamaha YVC-300 Pro")
    assert dw.speaker_ok("YVC-300", audio) is True
    assert dw.speaker_ok("YVC-300", _FakeAudio("pro-audio", "内部オーディオ")) is False


def test_the_speaker_is_repaired_by_pro_audio_and_the_default_output() -> None:
    """11/14 の形：使い方が off で、既定の出口が内部オーディオ。出口はあとから現れる。"""
    audio = _FakeAudio("off", "内部オーディオ", appear_after=2)
    assert dw.repair_speaker("YVC-300", audio, sleep=lambda s: None) is True
    assert ["wpctl", "set-profile", "49", "1"] in audio.calls
    assert ["wpctl", "set-default", "59"] in audio.calls
    assert dw.speaker_ok("YVC-300", audio) is True


def test_when_the_profile_is_already_pro_audio_only_the_default_is_set() -> None:
    audio = _FakeAudio("pro-audio", "内部オーディオ")
    assert dw.repair_speaker("YVC-300", audio, sleep=lambda s: None) is True
    assert not [c for c in audio.calls if c[:2] == ["wpctl", "set-profile"]]


def test_a_speaker_whose_output_never_appears_is_not_repaired() -> None:
    audio = _FakeAudio("off", "内部オーディオ", appear_after=99)
    assert dw.repair_speaker("YVC-300", audio, sleep=lambda s: None) is False
    assert not [c for c in audio.calls if c[:2] == ["wpctl", "set-default"]]


def test_a_missing_device_is_not_repaired() -> None:
    audio = _FakeAudio("pro-audio", "内部オーディオ")
    assert dw.repair_speaker("つながっていない機器", audio, sleep=lambda s: None) is False


# ── 見張りの 1 回（確かめて、直して、確かめ直す）─────────────────────────────────


class _Item:
    """確かめと直し方の偽物。`fixed_by_repair` なら直したあとは使える。"""

    def __init__(self, name: str, ok: bool, *, fixed_by_repair: bool = True) -> None:
        self.name = name
        self._ok = ok
        self._fix = fixed_by_repair
        self.repaired = 0

    def check(self) -> "bool | None":
        return self._ok

    def repair(self) -> None:
        self.repaired += 1
        if self._fix:
            self._ok = True


def test_one_round_repairs_what_is_broken_and_logs_info_or_warning(caplog) -> None:
    ok = _Item("スピーカー", True)
    fixable = _Item("カメラ", False)
    stuck = _Item("マイク", False, fixed_by_repair=False)
    with caplog.at_level(logging.INFO, logger="familiar_agent.io.device_watch"):
        res = asyncio.run(dw.check_once([ok, fixable, stuck]))
    assert res.repaired == ["カメラ"] and res.failed == ["マイク"]
    assert ok.repaired == 0 and fixable.repaired == 1 and stuck.repaired == 1
    infos = [r for r in caplog.records if r.levelno == logging.INFO and "直した" in r.getMessage()]
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(infos) == 1 and "カメラ" in infos[0].getMessage()
    assert len(warns) == 1 and "マイク" in warns[0].getMessage()


def test_an_item_that_is_not_watched_is_skipped() -> None:
    """設定が無い・聞き取りを止めているときは None（確かめない）。"""
    off = _Item("カメラ", None)  # type: ignore[arg-type]
    res = asyncio.run(dw.check_once([off]))
    assert res == CheckResult() and off.repaired == 0


def test_a_check_that_raises_does_not_stop_the_others(caplog) -> None:
    class Boom(_Item):
        def check(self):
            raise RuntimeError("壊れた")

    other = _Item("カメラ", False)
    res = asyncio.run(dw.check_once([Boom("スピーカー", True), other]))
    assert res.failed == ["スピーカー"] and res.repaired == ["カメラ"]


def test_the_watch_waits_first_then_paces_by_the_interval() -> None:
    waits: list[float] = []

    async def fake_sleep(s: float) -> None:
        waits.append(s)
        if len(waits) >= 4:
            raise asyncio.CancelledError

    item = _Item("スピーカー", True)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(dw.run([item], cfg=_CFG, sleep=fake_sleep))
    assert waits == [60.0, 300.0, 600.0, 1200.0]


# ── カメラ（§3.3）──────────────────────────────────────────────────────────


class _FakeCap:
    def __init__(self, cam, opened: bool) -> None:
        self._cam = cam
        self._opened = opened

    def isOpened(self) -> bool:  # noqa: N802  cv2 の名前
        return self._opened

    def read(self):
        import numpy as np

        self._cam._running = False  # 1 枚読んだら仕事を終える
        return True, np.zeros((2, 2, 3), dtype="uint8")

    def release(self) -> None:
        pass


def _camera(monkeypatch, *, opened: bool):
    from familiar_agent.tools import camera as cam_mod

    cam = cam_mod.CameraTool("192.0.2.1", "u", "p", 2020)
    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", lambda *a, **k: _FakeCap(cam, opened))
    return cam


def test_the_camera_remembers_when_it_last_got_a_picture(monkeypatch) -> None:
    import time

    cam = _camera(monkeypatch, opened=True)
    assert cam.frame_age() is None  # 仕事を立てていなければ確かめない
    cam._started_at = time.monotonic() - 100.0
    cam._running = True
    cam._capture_loop()
    age = cam.frame_age()
    assert age is not None and age < 5.0


def test_a_camera_that_failed_to_open_grows_old_from_its_start(monkeypatch) -> None:
    """11/14 の形：最初に開けないと仕事が終わる。古さは仕事を立てた時刻から数える。"""
    import time

    cam = _camera(monkeypatch, opened=False)
    cam._started_at = time.monotonic() - 100.0
    cam._running = True
    cam._capture_loop()
    assert cam._running is False
    age = cam.frame_age()
    assert age is not None and age >= 100.0


def test_reopen_restarts_only_a_finished_capture_thread(monkeypatch) -> None:
    import threading

    from familiar_agent.tools.camera import CameraTool

    cam = CameraTool("192.0.2.1", "u", "p", 2020)
    started: list[int] = []
    monkeypatch.setattr(CameraTool, "start", lambda self: started.append(1))
    cam._thread = None
    assert cam.reopen() is True
    assert started == [1]

    alive = threading.Event()
    t = threading.Thread(target=alive.wait, daemon=True)
    t.start()
    cam._thread = t
    assert cam.reopen() is False  # 生きている仕事のそばに二つ目を立てない
    alive.set()
    t.join()
    assert started == [1]


def test_a_fresh_item_waits_for_new_data_after_the_repair() -> None:
    """開き直したあと、新しい映像が来るまで待つ。来なければ直せなかったと分かる。"""
    state = {"age": 999.0, "ticks": 0}

    async def sleep(s: float) -> None:
        state["ticks"] += 1
        if state["ticks"] == 3:
            state["age"] = 0.5  # 3 秒目に新しい映像が来た

    item = dw.Fresh(
        "カメラ", lambda: state["age"], lambda: None, stale_sec=60.0, settle_sec=30.0, sleep=sleep
    )
    res = asyncio.run(dw.check_once([item]))
    assert res.repaired == ["カメラ"] and state["ticks"] == 3

    state.update(age=999.0, ticks=0)
    never = dw.Fresh(
        "カメラ", lambda: 999.0, lambda: None, stale_sec=60.0, settle_sec=5.0, sleep=sleep
    )
    assert asyncio.run(dw.check_once([never])).failed == ["カメラ"]


def test_a_reopened_camera_that_still_cannot_open_stays_old(monkeypatch) -> None:
    """立て直しで古さを 0 に戻さない（開けないままなら、確かめ直しで直せなかったと分かる）。"""
    import time

    from familiar_agent.tools.camera import CameraTool

    cam = CameraTool("192.0.2.1", "u", "p", 2020)
    monkeypatch.undo()  # 共通の守り（start を何もしない）を外し、本物の start の時刻の扱いを見る
    monkeypatch.setattr(CameraTool, "_capture_loop", lambda self: None)
    cam.start()
    first = cam._started_at
    cam._started_at = time.monotonic() - 100.0
    cam.reopen()
    age = cam.frame_age()
    assert first is not None and age is not None and age >= 100.0


# ── マイク（§3.2）──────────────────────────────────────────────────────────


def test_the_mic_age_counts_from_the_last_audio_even_while_speaking(monkeypatch) -> None:
    """声を出しているあいだ聞き取りへは渡さないが、音が届いていることは数える。"""
    import time

    from familiar_agent.realtime_stt_session import RealtimeSttController, RealtimeSttSession

    class _Guard:
        speaking = True

    s = RealtimeSttSession("", voice_guard=_Guard())  # type: ignore[arg-type]
    c = RealtimeSttController(s)
    assert c.mic_audio_age() is None  # 聞き取りを止めているときは確かめない
    s._mic_wanted = True
    s._mic_started_at = time.monotonic() - 100.0
    age = c.mic_audio_age()
    assert age is not None and age >= 100.0  # まだ音が来ていなければ、始めてからの秒
    asyncio.run(s._send_audio(b"\x00\x00"))
    age = c.mic_audio_age()
    assert age is not None and age < 5.0


# ── 見張る機器の組み立て（§3.4）───────────────────────────────────────────────


def test_build_items_watches_only_what_is_configured() -> None:
    class _Cam:
        def frame_age(self):
            return 1.0

        def reopen(self):
            pass

    names = [
        i.name
        for i in dw.build_items(
            device_name=lambda: "YVC-300",
            mic_age=lambda: 1.0,
            mic_restart=lambda: None,
            camera=_Cam(),
            cfg=_CFG,
        )
    ]
    assert names == ["スピーカー", "マイク", "カメラ"]
    names = [
        i.name
        for i in dw.build_items(
            device_name=lambda: "",
            mic_age=lambda: None,
            mic_restart=lambda: None,
            camera=None,
            cfg=_CFG,
        )
    ]
    assert names == [
        "スピーカー",
        "マイク",
    ]  # 名前の設定が無い、聞き取りを止めているときは確かめの中で None


def test_a_mic_opened_on_another_device_is_not_usable() -> None:
    """YAMAHA が見つからず既定の入力へ落ちたら、無音の音が届いていても使えないとする。"""
    from familiar_agent.realtime_stt_session import RealtimeSttSession

    class _Guard:
        speaking = False

    class _Mic:
        device_name = "HDA Intel PCH: ALC897 Analog"

    s = RealtimeSttSession("", voice_guard=_Guard())  # type: ignore[arg-type]
    s._mic_wanted = True
    s._mic_capture = _Mic()  # type: ignore[assignment]
    asyncio.run(s._send_audio(b"\x00\x00"))
    assert s.mic_audio_age("YVC-300") == float("inf")
    _Mic.device_name = "Yamaha YVC-300: USB Audio"
    age = s.mic_audio_age("YVC-300")
    assert age is not None and age < 5.0


def test_a_restart_that_fails_keeps_the_mic_watched(monkeypatch) -> None:
    """始め直しの途中で落ちても、次の回にまた確かめて始め直す（止めたのと区別する）。"""
    from familiar_agent.realtime_stt_session import RealtimeSttSession

    class _Guard:
        speaking = False

    s = RealtimeSttSession("", voice_guard=_Guard())  # type: ignore[arg-type]
    s._loop = object()  # type: ignore[assignment]
    s._committed_queue = object()  # type: ignore[assignment]
    s._mic_started_at = 0.0

    async def fake_stop():
        s._mic_wanted = False

    async def fail_start(loop, queue):
        raise RuntimeError("担い手を読み込めない")

    monkeypatch.setattr(s, "stop", fake_stop)
    monkeypatch.setattr(s, "start", fail_start)
    with pytest.raises(RuntimeError):
        asyncio.run(s.restart(reason="device-watch"))
    assert s.mic_audio_age() is not None


def test_a_camera_thread_still_running_is_not_counted_as_repaired() -> None:
    """立て直さなかったら（生きている仕事がある）、待たずに確かめ直す。"""
    waited: list[float] = []

    async def sleep(s: float) -> None:
        waited.append(s)

    item = dw.Fresh(
        "カメラ", lambda: 999.0, lambda: False, stale_sec=60.0, settle_sec=30.0, sleep=sleep
    )
    assert asyncio.run(dw.check_once([item])).failed == ["カメラ"]
    assert waited == []


def test_the_sink_is_looked_up_again_after_the_profile_changes() -> None:
    """前の使い方の出口は消えるので、使い方を変える前の番号を既定にしない。"""
    audio = _FakeAudio("off", "内部オーディオ", appear_after=0)
    real = audio.__call__

    def with_old_sink(cmd):
        out = real(cmd)
        if cmd == ["pw-dump"] and audio.profile == "off":
            objs = json.loads(out)
            objs.append(
                {
                    "id": 77,
                    "type": "PipeWire:Interface:Node",
                    "info": {
                        "props": {
                            "media.class": "Audio/Sink",
                            "node.description": "Yamaha YVC-300 モノ",
                        }
                    },
                }
            )
            return json.dumps(objs)
        return out

    assert dw.repair_speaker("YVC-300", with_old_sink, sleep=lambda s: None) is True
    assert ["wpctl", "set-default", "77"] not in audio.calls
    assert ["wpctl", "set-default", "59"] in audio.calls
