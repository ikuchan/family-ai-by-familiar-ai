"""在席の情報源を二層に分ける（在/不在＝センサ、誰か＝PMM）。

用語一覧の二層に対し、`scan_presence` はこれまで PMM（InsightFace の照合）だけを読んでいた。
照合は登録が要るので、登録が済むまで**在席が一切動かない**（実機で退室イベントが一度も
通っていなかった）。

在/不在は `PresenceSensor`（YOLO・登録不要）が持つ。名前は分かるときだけ PMM から取る。
分からなければ「誰か」として扱い、**居ることは伝える**。名前が無いことと、誰も居ないことは
別である。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from familiar_agent.loop.tonic import Tonic


def _tonic(*, names=(), occupied=False):
    agent = MagicMock()
    agent._pmm.presence_status = MagicMock(return_value=[{"name": n} for n in names])
    sensor = MagicMock()
    sensor.room_occupied = MagicMock(return_value=occupied)
    ip = MagicMock()
    t = Tonic(ip, presence=sensor)
    t._agent = agent
    t._present_names = set()  # 初回走査の扱いを飛ばす
    return t, ip


def test_someone_unidentified_still_counts_as_present():
    t, ip = _tonic(names=(), occupied=True)
    t.scan_presence()
    kinds = [c.args[0] for c in ip.note_device.call_args_list]
    assert "入室" in kinds


def test_a_known_name_is_used_when_the_face_matched():
    t, ip = _tonic(names=("パパ",), occupied=True)
    t.scan_presence()
    assert "パパ" in ip.note_device.call_args_list[0].args[1]


def test_an_empty_room_pushes_nothing():
    t, ip = _tonic(names=(), occupied=False)
    t.scan_presence()
    ip.note_device.assert_not_called()


def test_leaving_is_detected_from_the_sensor_alone():
    t, ip = _tonic(names=(), occupied=False)
    t._present_names = {"誰か"}
    t.scan_presence()
    assert [c.args[0] for c in ip.note_device.call_args_list] == ["退室"]


def test_a_name_appearing_later_does_not_double_count_the_same_person():
    """先に「誰か」で入室し、後から顔が照合できた場合。

    素朴に差分を取ると「誰か が居なくなった」「パパ が来た」の2件が飛ぶ。実際には
    同じ人がそこに居続けているだけで、退室は起きていない。
    """
    t, ip = _tonic(names=("パパ",), occupied=True)
    t._present_names = {"誰か"}
    t.scan_presence()
    kinds = [c.args[0] for c in ip.note_device.call_args_list]
    assert "退室" not in kinds


def test_without_a_sensor_the_old_source_still_works():
    # カメラが無い構成では PMM（`/speaker` の自己申告を含む）だけで動く。
    agent = MagicMock()
    agent._pmm.presence_status = MagicMock(return_value=[{"name": "たいき"}])
    ip = MagicMock()
    t = Tonic(ip)
    t._agent = agent
    t._present_names = set()
    t.scan_presence()
    assert "たいき" in ip.note_device.call_args_list[0].args[1]


# ── 在席表の失効（2026-09-17）────────────────────────────────────────────────
#
# `/speaker パパ` は在席表（PMM）に入るが、出る口が無かった（`PresenceWatcher` は未起動・
# `note_person_left` は呼ばれない）。センサが「誰も居ない」を 1 分見続けたら在席表を空にする。
# 話者の指定（誰が話しているか）は残す——顔と声の登録までは `/speaker` が唯一の手がかり。


def _tonic_with_table(*, names, occupied):
    t, ip = _tonic(names=names, occupied=occupied)
    # 失効が回すのは `present_keys()`——**名前の分からない在席者の札も含む**（出-am・2026-09-22）。
    # `get_present_ids()` は観測の `participants` になる口で札を含まないので、そちらを回すと
    # 名前の無い在席者だけが残ったときに 60 秒で畳めない。
    t._agent._pmm.present_keys = MagicMock(return_value=[f"id:{n}" for n in names])
    t._agent.config.presence_expire_sec = 60.0
    return t, ip


def test_the_presence_table_expires_after_a_minute_of_nobody(monkeypatch):
    t, ip = _tonic_with_table(names=("パパ",), occupied=False)
    t._present_names = {"パパ"}
    clock = iter([1000.0, 1061.0])
    monkeypatch.setattr("familiar_agent.loop.tonic.time.time", lambda: next(clock))
    t.scan_presence()  # 誰も居ない、を見始めた
    t._agent._pmm.mark_absent.assert_not_called()
    t.scan_presence()  # 61 秒後
    t._agent._pmm.mark_absent.assert_called_once_with("id:パパ")


def test_fifty_nine_seconds_is_not_enough(monkeypatch):
    t, ip = _tonic_with_table(names=("パパ",), occupied=False)
    t._present_names = {"パパ"}
    clock = iter([1000.0, 1059.0])
    monkeypatch.setattr("familiar_agent.loop.tonic.time.time", lambda: next(clock))
    t.scan_presence()
    t.scan_presence()
    t._agent._pmm.mark_absent.assert_not_called()


def test_seeing_someone_resets_the_count(monkeypatch):
    t, ip = _tonic_with_table(names=("パパ",), occupied=False)
    t._present_names = {"パパ"}
    clock = iter([1000.0, 1030.0, 1070.0])
    monkeypatch.setattr("familiar_agent.loop.tonic.time.time", lambda: next(clock))
    t.scan_presence()
    t._presence.room_occupied = MagicMock(return_value=True)
    t.scan_presence()  # 30 秒後に人を見た → 数え直し
    t._presence.room_occupied = MagicMock(return_value=False)
    t.scan_presence()  # そこから 40 秒 → まだ
    t._agent._pmm.mark_absent.assert_not_called()


def test_marking_absent_keeps_the_speaker():
    from familiar_agent.person_memory_manager import PersonMemoryManager

    pmm = PersonMemoryManager.__new__(PersonMemoryManager)
    import threading

    pmm._lock = threading.Lock()
    pmm._present = {"id:パパ": MagicMock()}
    pmm._speaker_id = "id:パパ"
    pmm.mark_absent("id:パパ")
    assert pmm.get_present_ids() == []
    assert pmm._speaker_id == "id:パパ"


def test_a_fresh_speaker_command_restarts_the_expiry_clock(monkeypatch):
    """`/speaker` を打った 0.5 秒後に「誰も居ないが 340 秒」で失効した（2026-09-18 12:32 実機・知-s）。"""
    t, ip = _tonic_with_table(names=("パパ",), occupied=False)
    t._present_names = {"パパ"}
    t._agent._speaker_set_at = 1300.0  # 打ったのは 1300
    clock = iter([1000.0, 1301.0, 1350.0, 1361.0])
    monkeypatch.setattr("familiar_agent.loop.tonic.time.time", lambda: next(clock))
    t.scan_presence()  # 1000：誰も居ない、を見始めた
    t.scan_presence()  # 1301：打った直後——失効しない
    t._agent._pmm.mark_absent.assert_not_called()
    t.scan_presence()  # 1350：打ってから 50 秒——まだ
    t._agent._pmm.mark_absent.assert_not_called()
    t.scan_presence()  # 1361：打ってから 61 秒
    t._agent._pmm.mark_absent.assert_called_once()
