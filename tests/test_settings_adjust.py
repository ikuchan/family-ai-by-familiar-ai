"""層 3「設定値を調整する」（記-a-に・2026-09-14）。

計測ログの集計（数字）から、登録した規則で値を 1 晩 1 値 1 刻み動かす。範囲の端で止まる。
機械の規則：窓 n（端か外の参照が 20% 以上なら +1、続きがあるのに端も外も参照されなければ −1）、
内部状態の境目（分位を取り直す）、`diffuse_far_share`（参照された側へ 0.1）。LLM の規則：
`arbiter_timeout_sec`（集計の数字を渡し、提案を受ける）。動かしたら `内省` の記録と計測ログに残す。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent import config_overrides as co
from familiar_agent.loop import rest_settings as rs


def _reset():
    co._delete_all()
    co.clear_cache()


def test_the_window_grows_when_the_edge_or_beyond_is_referenced_often():
    _reset()
    changes = rs.adjust_window({"続き": 5, "端を参照": 1, "外を参照": 1, "端か外の割合": 0.4})
    assert changes == [
        ("MemoryConfig.recent_exchanges_main", 6, 7),
        ("MemoryConfig.recent_exchanges_arbiter", 3, 4),
    ]


def test_the_window_shrinks_when_continuations_never_reach_the_edge():
    _reset()
    changes = rs.adjust_window({"続き": 5, "端を参照": 0, "外を参照": 0, "端か外の割合": 0.0})
    assert changes == [
        ("MemoryConfig.recent_exchanges_main", 6, 5),
        ("MemoryConfig.recent_exchanges_arbiter", 3, 2),
    ]


def test_the_window_stays_without_continuations_and_stops_at_the_range():
    _reset()
    assert rs.adjust_window({"続き": 0, "端を参照": 0, "外を参照": 0, "端か外の割合": 0.0}) == []
    co.save_override("MemoryConfig.recent_exchanges_main", 10)
    co.save_override("MemoryConfig.recent_exchanges_arbiter", 10)
    co.clear_cache()
    assert rs.adjust_window({"続き": 5, "端を参照": 2, "外を参照": 1, "端か外の割合": 0.6}) == []
    _reset()


def test_inner_state_boundaries_follow_the_measured_quantiles_one_step_at_a_time():
    _reset()
    summary = {"P": {"p10": 0.10, "p30": 0.12, "p70": 0.40, "p90": 0.50}}
    changes = rs.adjust_inner_state(summary)
    # p70 は 0.25 → 0.30（刻み 0.05 で 1 段だけ・目標 0.40 に近づく）、p90 は 0.35 → 0.40、p30 は 0.10 → 0.12 に届かず（刻み未満は動かさない）
    assert ("InnerStateConfig.mood_p_p70", 0.25, 0.30) in changes
    assert ("InnerStateConfig.mood_p_p90", 0.35, 0.40) in changes
    assert not any(c[0] == "InnerStateConfig.mood_p_p30" for c in changes)


def test_far_share_moves_toward_the_referenced_order():
    _reset()
    assert rs.adjust_far_share({"遠い": 5, "掘り": 1}) == [
        ("MemoryConfig.diffuse_far_share", 0.5, 0.6)
    ]
    assert rs.adjust_far_share({"遠い": 1, "掘り": 5}) == [
        ("MemoryConfig.diffuse_far_share", 0.5, 0.4)
    ]
    assert rs.adjust_far_share({"遠い": 0, "掘り": 0}) == []


def test_the_timeout_is_proposed_by_the_llm_from_numbers_only():
    _reset()
    a = MagicMock()
    a.backend = AsyncMock()
    a.backend.complete = AsyncMock(
        return_value='{"arbiter_timeout_sec": 5.5, "reason": "p90 が 4.8 秒で時間切れ 12%"}'
    )
    changes = asyncio.run(
        rs.adjust_timeout(
            a,
            {
                "件数": 50,
                "中央": 1.0,
                "p90": 4.8,
                "最大": 5.0,
                "時間切れ": 6,
                "時間切れの割合": 0.12,
            },
        )
    )
    assert changes == [("AgentConfig.arbiter_timeout_sec", 5.0, 5.5)]
    prompt = a.backend.complete.call_args.args[0]
    assert (
        "p90" in prompt and "4.8" in prompt and "調停 秒=" not in prompt
    )  # 数字だけ、生の行は無い


def test_the_llm_proposal_is_clamped_to_one_step_and_the_range():
    _reset()
    a = MagicMock()
    a.backend = AsyncMock()
    a.backend.complete = AsyncMock(return_value='{"arbiter_timeout_sec": 9.0}')
    assert asyncio.run(
        rs.adjust_timeout(
            a, {"件数": 5, "中央": 1, "p90": 1, "最大": 1, "時間切れ": 0, "時間切れの割合": 0}
        )
    ) == [("AgentConfig.arbiter_timeout_sec", 5.0, 5.5)]
    a.backend.complete = AsyncMock(return_value="???")
    assert (
        asyncio.run(
            rs.adjust_timeout(
                a, {"件数": 5, "中央": 1, "p90": 1, "最大": 1, "時間切れ": 0, "時間切れの割合": 0}
            )
        )
        == []
    )


def test_apply_writes_the_overrides_and_records_the_changes(tmp_path):
    from familiar_agent.core import measure

    _reset()
    measure.setup(base_dir=tmp_path)
    a = MagicMock()
    a._memory.save_async_with_id = AsyncMock(return_value=("obs", True))
    a._observation_perspective = MagicMock(return_value={})
    asyncio.run(rs.apply(a, [("MemoryConfig.recent_exchanges_main", 6, 7)]))
    co.clear_cache()
    assert co.load_overrides()["MemoryConfig.recent_exchanges_main"] == 7
    text = (tmp_path / "rest_logs" / "measure.log").read_text(encoding="utf-8")
    assert "設定値 名前=MemoryConfig.recent_exchanges_main 前=6 後=7" in text
    assert a._memory.save_async_with_id.await_count == 1
    _reset()
