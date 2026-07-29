"""REST 内省の起動（#2・順1＝骨格）。

設計は `直近の進め方と進捗` v0.14 が定める折衷型で、起動は T の純粋欠乏発火（日次）。
1パスは 読み込み → 蒸留 → open 棚卸し → Config 自己調整 で、圧縮系は量ベース。
**ここで作るのは起動条件と骨格だけ**で、パスの中身は後続で足す。

在/不在は `PresenceSensor`（YOLO・登録が要らない）で見る。`_social_presence_permission()`
は PMM（顔の照合）と直近の発話しか見ないので使わない（#15 で判明した欠陥）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from familiar_agent.core.drive_dynamics import DriveFiring
from familiar_agent.drive_register import AiDrivers
from familiar_agent.loop.tonic import Tonic

_REST = DriveFiring(seeking=False, rest=True, bond=False, safety=False, esteem=False)
_SEEKING = DriveFiring(seeking=True, rest=False, bond=False, safety=False, esteem=False)


def _ip():
    ip = MagicMock()
    ip.push_affect = MagicMock()
    return ip


def _sensor(*, occupied: bool):
    s = MagicMock()
    s.room_occupied = MagicMock(return_value=occupied)
    return s


def _run_until(firing, *, presence, predicate, timeout_ticks: int = 400):
    """発火を1回起こし、`predicate` が真になるまで待つ（T は 0.01 秒周期）。"""
    ip = _ip()
    rest_pass = AsyncMock(return_value="内省した")

    async def scenario():
        with patch("familiar_agent.loop.tonic.step_drives",
                   new=AsyncMock(return_value=(firing, AiDrivers()))), \
             patch("familiar_agent.loop.tonic.run_rest_pass", new=rest_pass):
            t = Tonic(ip, agent=MagicMock(), period=0.01, presence=presence)
            t.start()
            for _ in range(timeout_ticks):
                if predicate(ip, rest_pass):
                    break
                await asyncio.sleep(0.005)
            await t.close()

    asyncio.run(scenario())
    return ip, rest_pass


def test_rest_starts_the_introspection_pass_when_nobody_is_present():
    """誰も居ないときの REST 発火は、自発ターンではなく内省パスへ入る。"""
    ip, rest_pass = _run_until(
        _REST, presence=_sensor(occupied=False),
        predicate=lambda ip, rp: rp.await_count > 0,
    )
    assert rest_pass.await_count == 1
    assert not ip.push_affect.called        # 人へ話しかける自発ターンにはしない


def test_rest_still_speaks_when_someone_is_present():
    """誰か居るときの REST 発火は従来どおり。「休みたい」と伝えるのは自然な振る舞い。"""
    ip, rest_pass = _run_until(
        _REST, presence=_sensor(occupied=True),
        predicate=lambda ip, rp: ip.push_affect.call_count > 0,
    )
    assert ip.push_affect.call_args.args[0] == "REST"
    assert rest_pass.await_count == 0


def test_rest_speaks_when_there_is_no_presence_sensor():
    """センサが無い構成（カメラ無し）では従来どおり。

    「センサが無い」を「誰も居ない」と扱うと、カメラの無い環境で REST が常に内省へ
    落ちて、人が居ても話しかけなくなる。
    """
    ip, rest_pass = _run_until(
        _REST, presence=None,
        predicate=lambda ip, rp: ip.push_affect.call_count > 0,
    )
    assert ip.push_affect.call_args.args[0] == "REST"
    assert rest_pass.await_count == 0


def test_other_drives_are_unaffected_by_absence():
    """REST 以外は、誰も居なくても従来どおり QA へ積む（内省は REST の役目）。"""
    ip, rest_pass = _run_until(
        _SEEKING, presence=_sensor(occupied=False),
        predicate=lambda ip, rp: ip.push_affect.call_count > 0,
    )
    assert ip.push_affect.call_args.args[0] == "SEEKING"
    assert rest_pass.await_count == 0


def test_rest_pass_records_what_it_did():
    """内省パスは、何をしたかを O に残す（記録が無いと回ったことを確かめられない）。"""
    from familiar_agent.loop.rest import run_rest_pass

    agent = MagicMock()
    agent._memory.save_async_with_id = AsyncMock(return_value=("obs1", True))
    agent._observation_perspective = MagicMock(return_value={})

    asyncio.run(run_rest_pass(agent))

    assert agent._memory.save_async_with_id.await_count == 1
    content = agent._memory.save_async_with_id.call_args.args[0]
    assert "内省" in content
