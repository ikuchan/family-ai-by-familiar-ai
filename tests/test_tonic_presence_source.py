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
    agent._pmm.presence_status = MagicMock(
        return_value=[{"name": n} for n in names])
    sensor = MagicMock()
    sensor.room_occupied = MagicMock(return_value=occupied)
    ip = MagicMock()
    t = Tonic(ip, presence=sensor)
    t._agent = agent
    t._present_names = set()          # 初回走査の扱いを飛ばす
    return t, ip


def test_someone_unidentified_still_counts_as_present():
    t, ip = _tonic(names=(), occupied=True)
    t.scan_presence()
    kinds = [c.args[0] for c in ip.push_device.call_args_list]
    assert "入室" in kinds


def test_a_known_name_is_used_when_the_face_matched():
    t, ip = _tonic(names=("パパ",), occupied=True)
    t.scan_presence()
    assert "パパ" in ip.push_device.call_args_list[0].args[1]


def test_an_empty_room_pushes_nothing():
    t, ip = _tonic(names=(), occupied=False)
    t.scan_presence()
    ip.push_device.assert_not_called()


def test_leaving_is_detected_from_the_sensor_alone():
    t, ip = _tonic(names=(), occupied=False)
    t._present_names = {"誰か"}
    t.scan_presence()
    assert [c.args[0] for c in ip.push_device.call_args_list] == ["退室"]


def test_a_name_appearing_later_does_not_double_count_the_same_person():
    """先に「誰か」で入室し、後から顔が照合できた場合。

    素朴に差分を取ると「誰か が居なくなった」「パパ が来た」の2件が飛ぶ。実際には
    同じ人がそこに居続けているだけで、退室は起きていない。
    """
    t, ip = _tonic(names=("パパ",), occupied=True)
    t._present_names = {"誰か"}
    t.scan_presence()
    kinds = [c.args[0] for c in ip.push_device.call_args_list]
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
    assert "たいき" in ip.push_device.call_args_list[0].args[1]
