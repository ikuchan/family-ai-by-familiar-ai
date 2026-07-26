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
import contextlib
import logging
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.tonic import Tonic


@contextlib.contextmanager
def _capture():
    """`tonic` のログを拾う。"""
    records: list[str] = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("familiar_agent.loop.tonic")
    handler, old = _H(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old)


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


def test_presence_scan_leaves_a_trace_even_when_nothing_changes():
    # イベントが出ないとき、「T が回っていない」のか「誰も居ない」のかを区別できる
    # 必要がある。初回走査で見えているものを残し、変化したときは前後を残す。
    ip = _ip()
    t = Tonic(ip, agent=_agent_with(_rows(), _rows("パパ")))
    with _capture() as logs:
        t.scan_presence()          # 初回＝誰も居ない
        t.scan_presence()          # パパが来た
    joined = "\n".join(logs)
    assert "初回走査" in joined and "誰も居ない" in joined
    assert "在席の変化" in joined and "パパ" in joined


def test_driver_waits_only_on_completions_while_a_lookup_is_in_flight():
    # 調査中に情動や人の出入りで別の連鎖を始めると、1つの求めの途中に別の話が割り込む。
    # QA・QD は消費せずキューに残す（取りこぼしではなく待たせるだけ）。
    from familiar_agent.loop.event_loop import InformationProcessing

    a = MagicMock()

    async def scenario():
        ip = InformationProcessing(a)
        ip._iterate = AsyncMock(return_value="")
        ip._begin_affect = AsyncMock(return_value=None)
        ip._begin_device = AsyncMock(return_value=None)
        ip._inflight = 1                      # 調査が飛んでいる
        ip._ensure_driver()
        ip.push_affect("SEEKING", "なにか気になる")
        ip.push_device("入室", "パパ が来た")
        await asyncio.sleep(0.05)
        sizes = (ip._affect_queue.qsize(), ip._device_queue.qsize())
        await ip.close()
        return ip, sizes

    ip, sizes = asyncio.run(scenario())
    assert sizes == (1, 1)                    # どちらも消費されず残っている
    ip._begin_affect.assert_not_awaited()
    ip._begin_device.assert_not_awaited()


def test_held_speech_flows_into_w_with_when_it_was_wanted():
    # 保留していた発話は MI の content へ差し込まない（保留 O は想起でも W に上がるので
    # 二重になる）。W に「いつ・何を言いたかったか」として流し、言葉の組み立ては任せる。
    import datetime as _dt

    from familiar_agent.backend import ToolCall
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent, _turn

    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "おかえり"})])])
    created = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=7)
    a._pending_store.list_active = MagicMock(return_value=[{
        "id": "p1", "observation_id": "obs-held", "created_at": created,
        "content": "話したかったが、聞く相手が居なかった：こんばんは。",
    }])
    a._pending_store.freshness_score = MagicMock(return_value=1.0)
    a._pending_store.is_expired = MagicMock(return_value=False)

    async def scenario():
        ip = InformationProcessing(a)
        await ip._begin_device("入室", "パパ が来た", True)
        await ip.close()

    asyncio.run(scenario())
    system = "\n".join(a.backend.stream_turn.call_args.kwargs["system"])
    assert "聞く相手が居ないあいだに話したかったこと" in system
    assert "約7時間前" in system                     # 経過時間
    assert "こんばんは" in system
    a._pending_store.delete.assert_called_once_with("p1")
    # 配ったら元の O も閉じる（想起で上がり続けて蒸し返さないように）。
    assert ("obs-held",) == a._memory.mark_superseded.call_args.args[:1]

    # MI の content には差し込まない。
    device_mi = next(c.args[0] for c in a._memory.save_async_with_id.call_args_list
                     if c.kwargs.get("direction") == "機器")
    assert "こんばんは" not in device_mi


def test_releasing_held_speech_is_logged_with_its_count():
    # system プロンプトの全文は出していないので、件数を残さないと「載ったが触れられ
    # なかった」のか「そもそも載っていない」のかを区別できない（実機で、配られたのに
    # 発話がそれに触れなかった）。
    import datetime as _dt
    import logging

    from familiar_agent.loop.event_loop import InformationProcessing

    a = MagicMock()
    a._memory.save_async_with_id = AsyncMock(return_value=("obs1", True))
    a._observation_perspective = MagicMock(return_value={})
    a._pending_store.list_active = MagicMock(return_value=[{
        "id": "p1", "observation_id": "obs-held",
        "created_at": _dt.datetime.now(_dt.timezone.utc), "content": "こんばんは",
    }])
    a._pending_store.freshness_score = MagicMock(return_value=1.0)
    a._pending_store.is_expired = MagicMock(return_value=False)

    records: list[str] = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("familiar_agent.loop.event_loop")
    handler, old_level = _H(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)      # 既定は WARNING なので INFO が届かない
    try:
        ip = InformationProcessing(a)
        asyncio.run(ip._release_pending_speech())
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    assert any("保留を配る：1件" in m for m in records)
