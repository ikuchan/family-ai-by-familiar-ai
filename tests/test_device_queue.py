"""#11 段階3 スライス2：QD（DIFキュー＝機器）の結線。

QD に積むのは**人の出入り**（入室・退室）。動体そのものは「誰か」を判定できず、
イベントとしては粗すぎるので載せない（既定 off の `MOTION_WATCH` の既存経路は
そのまま・移設は #12）。

在席者の集合を見て前回との差分を取るのは T（時計を持つ唯一の側）。身元はいま PMM
（InsightFace）からしか取れず、これは #8 で二層（在/不在＝T、誰か＝I）へ整理する
暫定の層である。**QD に流れるイベントの形は身元の取得方法から独立**させてあるので、
#8 では情報源の付け替えだけで済む。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.tonic import Tonic


def _ip():
    ip = MagicMock()
    ip.push_device = MagicMock()
    return ip


def _agent_with(*presence_sequence):
    """presence_status() が呼ばれるたび次の在席者一覧を返す agent。"""
    a = MagicMock()
    a._pmm = MagicMock()
    a._pmm.presence_status = MagicMock(side_effect=list(presence_sequence))
    return a


def _rows(*names):
    return [{"person_id": n, "name": n, "confidence": 0.9, "is_speaker": False}
            for n in names]


def _scan(tonic, times):
    for _ in range(times):
        tonic.scan_presence()


def test_arrival_is_pushed_as_a_device_event():
    ip = _ip()
    t = Tonic(ip, agent=_agent_with(_rows(), _rows("たいき")))
    _scan(t, 2)
    kinds = [c.args[0] for c in ip.push_device.call_args_list]
    assert kinds == ["入室"]
    assert "たいき" in ip.push_device.call_args.args[1]


def test_departure_is_pushed_as_a_device_event():
    ip = _ip()
    t = Tonic(ip, agent=_agent_with(_rows("たいき"), _rows()))
    _scan(t, 2)
    assert [c.args[0] for c in ip.push_device.call_args_list] == ["退室"]
    assert "たいき" in ip.push_device.call_args.args[1]


def test_every_new_person_fires_even_when_someone_is_already_there():
    # 案B：既に誰か居るところへもう1人来ても立つ。
    ip = _ip()
    t = Tonic(ip, agent=_agent_with(_rows("パパ"), _rows("パパ", "たいき")))
    _scan(t, 2)
    assert [c.args[0] for c in ip.push_device.call_args_list] == ["入室"]


def test_no_event_while_the_same_people_stay():
    ip = _ip()
    t = Tonic(ip, agent=_agent_with(_rows("パパ"), _rows("パパ"), _rows("パパ")))
    _scan(t, 3)
    ip.push_device.assert_not_called()


def test_pending_speech_is_released_only_when_presence_rises_from_zero():
    # 保留は「聞く相手が居なかった」から溜まったもの。相手が現れた瞬間だけ配る。
    # 会話中に家族が増えるたび割り込ませない。
    ip = _ip()
    t = Tonic(ip, agent=_agent_with(_rows(), _rows("パパ"), _rows("パパ", "たいき")))
    _scan(t, 3)
    releases = [c.kwargs.get("release_pending") for c in ip.push_device.call_args_list]
    assert releases == [True, False]


def test_device_queue_wakes_the_driver():
    from familiar_agent.loop.event_loop import InformationProcessing

    a = MagicMock()
    a._memory = MagicMock()
    a._memory.save_async_with_id = AsyncMock(return_value=("obs1", True))
    a._observation_perspective = MagicMock(return_value={})

    async def scenario():
        ip = InformationProcessing(a)
        ip._iterate = AsyncMock(return_value="")
        ip._ensure_driver()
        ip.push_device("入室", "たいき が来た")
        for _ in range(200):
            if ip._iterate.await_count:
                break
            await asyncio.sleep(0.005)
        await ip.close()
        return ip

    ip = asyncio.run(scenario())
    ip._iterate.assert_awaited()
    assert ip._origin_kind == "機器"
