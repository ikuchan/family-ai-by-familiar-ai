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

import pytest

from familiar_agent.core.drive_dynamics import DriveFiring
from familiar_agent.drive_register import AiDrivers
from familiar_agent.loop.tonic import Tonic


@pytest.fixture(autouse=True)
def _no_layer_four():
    """層 4 は実装の docstring・`.env`・DB を読む。層 4 自身の試験は `test_rest_capabilities.py`。"""
    with patch(
        "familiar_agent.loop.rest.redefine_capabilities",
        new=AsyncMock(return_value="能力は見送った"),
    ):
        yield


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
        with (
            patch(
                "familiar_agent.loop.tonic.step_drives",
                new=AsyncMock(return_value=(firing, AiDrivers())),
            ),
            patch("familiar_agent.loop.tonic.run_rest_pass", new=rest_pass),
        ):
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
        _REST,
        presence=_sensor(occupied=False),
        predicate=lambda ip, rp: rp.await_count > 0,
    )
    assert rest_pass.await_count == 1
    assert not ip.push_affect.called  # 人へ話しかける自発ターンにはしない


def test_rest_still_speaks_when_someone_is_present():
    """誰か居るときの REST 発火は従来どおり。「休みたい」と伝えるのは自然な振る舞い。"""
    ip, rest_pass = _run_until(
        _REST,
        presence=_sensor(occupied=True),
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
        _REST,
        presence=None,
        predicate=lambda ip, rp: ip.push_affect.call_count > 0,
    )
    assert ip.push_affect.call_args.args[0] == "REST"
    assert rest_pass.await_count == 0


def test_other_drives_are_unaffected_by_absence():
    """REST 以外は、誰も居なくても従来どおり QA へ積む（内省は REST の役目）。"""
    ip, rest_pass = _run_until(
        _SEEKING,
        presence=_sensor(occupied=False),
        predicate=lambda ip, rp: ip.push_affect.call_count > 0,
    )
    assert ip.push_affect.call_args.args[0] == "SEEKING"
    assert rest_pass.await_count == 0


def test_rest_pass_folds_the_day_and_records_what_it_did():
    """内省パスは層 1 の畳み込み（記-a-ろ-は）を呼び、何をしたかを O に残す。"""
    from unittest.mock import patch

    from familiar_agent.loop.rest import run_rest_pass
    from familiar_agent.loop.rest_fold import FoldResult

    agent = MagicMock()
    agent._memory.save_async_with_id = AsyncMock(return_value=("obs1", True))
    agent._observation_perspective = MagicMock(return_value={})

    with patch(
        "familiar_agent.loop.rest.fold_since_last_rest",
        new=AsyncMock(
            return_value=FoldResult(materials=12, batches=1, written=3, folded=12, skipped=0)
        ),
    ) as fold:
        content = asyncio.run(run_rest_pass(agent))

    fold.assert_awaited_once()
    assert agent._memory.save_async_with_id.await_count == 1
    assert "12 件を畳" in content and "3" in content
    assert agent._memory.save_async_with_id.call_args.kwargs["direction"] == "内省"


def test_rest_pass_says_when_there_was_nothing_to_fold():
    from unittest.mock import patch

    from familiar_agent.loop.rest import run_rest_pass
    from familiar_agent.loop.rest_fold import FoldResult

    agent = MagicMock()
    agent._memory.save_async_with_id = AsyncMock(return_value=("obs1", True))
    agent._observation_perspective = MagicMock(return_value={})
    with patch(
        "familiar_agent.loop.rest.fold_since_last_rest",
        new=AsyncMock(
            return_value=FoldResult(materials=0, batches=0, written=0, folded=0, skipped=0)
        ),
    ):
        content = asyncio.run(run_rest_pass(agent))
    assert "畳むものが無" in content


def test_a_failing_fold_does_not_break_the_pass():
    """畳み込みが落ちても内省パスは記録を残して終わる（次の晩に持ち越す）。"""
    from unittest.mock import patch

    from familiar_agent.loop.rest import run_rest_pass

    agent = MagicMock()
    agent._memory.save_async_with_id = AsyncMock(return_value=("obs1", True))
    agent._observation_perspective = MagicMock(return_value={})
    with patch(
        "familiar_agent.loop.rest.fold_since_last_rest",
        new=AsyncMock(side_effect=RuntimeError("LLM が落ちた")),
    ):
        content = asyncio.run(run_rest_pass(agent))
    assert "畳めなかった" in content


def test_rest_pass_runs_layer_two_with_what_layer_one_wrote():
    """層 1 が書いた自己エピソードと関係のまとめを、そのまま層 2 の材料にする（記-a-へ）。"""
    from unittest.mock import patch

    from familiar_agent.loop.rest import run_rest_pass
    from familiar_agent.loop.rest_fold import FoldResult, Written
    from familiar_agent.loop.rest_self_image import Proposal

    agent = MagicMock()
    agent._memory.save_async_with_id = AsyncMock(return_value=("obs1", True))
    agent._observation_perspective = MagicMock(return_value={})
    written = (
        Written("ep-1", "day_summary", "今日は…"),
        Written("ps-1", "person_summary", "こうきは…"),
    )
    with (
        patch(
            "familiar_agent.loop.rest.fold_since_last_rest",
            new=AsyncMock(
                return_value=FoldResult(
                    materials=5, batches=1, written=2, folded=5, skipped=0, records=written
                )
            ),
        ),
        patch(
            "familiar_agent.loop.rest.update_self_image",
            new=AsyncMock(return_value=Proposal(image=MagicMock(), applied=True, changed=2)),
        ) as upd,
    ):
        content = asyncio.run(run_rest_pass(agent))
    materials = upd.await_args.args[1]
    assert [(m.obs_id, m.kind) for m in materials] == [
        ("ep-1", "day_summary"),
        ("ps-1", "person_summary"),
    ]
    assert "自己像" in content and "2 行" in content


def test_rest_pass_runs_layer_three_after_layer_two_and_reports_it():
    from unittest.mock import patch

    from familiar_agent.loop.rest import run_rest_pass
    from familiar_agent.loop.rest_fold import FoldResult
    from familiar_agent.loop.rest_self_image import Proposal

    agent = MagicMock()
    agent._memory.save_async_with_id = AsyncMock(return_value=("obs1", True))
    agent._observation_perspective = MagicMock(return_value={})
    order: list[str] = []
    with (
        patch(
            "familiar_agent.loop.rest.fold_since_last_rest",
            new=AsyncMock(side_effect=lambda a: (order.append("1"), FoldResult(0, 0, 0, 0, 0))[1]),
        ),
        patch(
            "familiar_agent.loop.rest.update_self_image",
            new=AsyncMock(
                side_effect=lambda a, m: (
                    order.append("2"),
                    Proposal(image=MagicMock(), applied=False, changed=0, reason="材料なし"),
                )[1]
            ),
        ),
        patch(
            "familiar_agent.loop.rest.adjust_settings",
            new=AsyncMock(side_effect=lambda a: (order.append("3"), 2)[1]),
        ),
    ):
        content = asyncio.run(run_rest_pass(agent))
    assert order == ["1", "2", "3"]
    assert "設定値を 2 件" in content


def test_rest_pass_runs_layer_four_last_and_tells_it_whether_the_self_image_changed():
    from unittest.mock import patch

    from familiar_agent.loop.rest import run_rest_pass
    from familiar_agent.loop.rest_fold import FoldResult
    from familiar_agent.loop.rest_self_image import Proposal

    agent = MagicMock()
    agent._memory.save_async_with_id = AsyncMock(return_value=("obs1", True))
    agent._observation_perspective = MagicMock(return_value={})
    order: list[str] = []
    seen: dict = {}

    async def _four(a, *, self_image_changed):
        order.append("4")
        seen["self_image_changed"] = self_image_changed
        return "要約を作り直した"

    with (
        patch(
            "familiar_agent.loop.rest.fold_since_last_rest",
            new=AsyncMock(side_effect=lambda a: (order.append("1"), FoldResult(0, 0, 0, 0, 0))[1]),
        ),
        patch(
            "familiar_agent.loop.rest.update_self_image",
            new=AsyncMock(
                side_effect=lambda a, m: (
                    order.append("2"),
                    Proposal(image=MagicMock(), applied=True, changed=1),
                )[1]
            ),
        ),
        patch(
            "familiar_agent.loop.rest.adjust_settings",
            new=AsyncMock(side_effect=lambda a: (order.append("3"), 0)[1]),
        ),
        patch("familiar_agent.loop.rest.redefine_capabilities", new=_four),
    ):
        content = asyncio.run(run_rest_pass(agent))
    assert order == ["1", "2", "3", "4"]
    assert seen["self_image_changed"] is True
    assert "要約を作り直した" in content
