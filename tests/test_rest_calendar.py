"""REST 層 1 の ③ 暦のまとめ（記-m 段 3・2026-10-01）。月と年の要約を書いて、記憶の木をつくる。

月が明けた晩に前の月を、年が明けた晩に前の年を、家族全体と人ごとに書く。過去のぶんは古い順に埋める。
1 晩に 6 本まで（本人の決定）。書き手はフルLLM（日次の畳み込みと同じ）。

- 家族全体の月：日ごとの要約がある月 ／ 人ごとの月：その人の人ごとのまとめがある月。いまの月は書かない。
- 年：いまの年より前で、その年の月の要約がそろっている年。
- 検査（空・読めない・字数の超過）に落ちたものは書かず、次の晩に持ち越す。
- 向き「月のまとめ／年のまとめ／人の月のまとめ／人の年のまとめ」、時刻はその月・年の終わり（UTC 23:59）、
  人ごとのものはその人の面（`participants=[pid]`）。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop import rest_calendar

UTC = timezone.utc
NOW = datetime(2026, 10, 1, 3, 0, tzinfo=UTC)
FAMILY = "## パパ\n\n- **名前**：雄輔\n- **呼び方**：パパ\n"


def _agent(months: dict, *, reply='{"summary": "ぼくはこの月、キャンプに行った。"}'):
    """`months[(kind, pid)]` に、その種類の要約がある月（YYYY-MM）を並べる。"""
    a = MagicMock()
    a._family_md = FAMILY
    a._pmm.find_person_id_by_name = MagicMock(
        side_effect=lambda name: "pid-papa" if name in ("雄輔", "パパ") else None
    )
    a._oif.tree_months = MagicMock(
        side_effect=lambda kind, person_id=None: months.get((kind, person_id), [])
    )
    a._oif.tree_children = MagicMock(
        side_effect=lambda node, person_id=None: [{"content": f"{node} の中身", "timestamp": NOW}]
    )
    a._oif.write = AsyncMock(side_effect=lambda mi, **kw: f"w-{mi.direction}-{mi.timestamp:%Y%m}")
    a._observation_perspective = MagicMock(return_value={"writer_id": "__self__"})
    a.backend.complete = AsyncMock(return_value=reply)
    return a


def _pending_of(a):
    return [(t.level, t.period, t.person_name) for t in rest_calendar.pending(a, now=NOW)]


def test_closed_months_are_listed_oldest_first_and_this_month_is_not():
    a = _agent({("day_summary", None): ["2026-10", "2026-08", "2026-09"]})
    assert _pending_of(a) == [("月", "2026-08", None), ("月", "2026-09", None)]


def test_a_month_already_summarized_is_skipped():
    a = _agent(
        {
            ("day_summary", None): ["2026-08", "2026-09"],
            ("month_summary", None): ["2026-08"],
        }
    )
    assert _pending_of(a) == [("月", "2026-09", None)]


def test_each_person_gets_their_own_months_after_the_family():
    a = _agent(
        {
            ("day_summary", None): ["2026-09"],
            ("person_summary", "pid-papa"): ["2026-09"],
        }
    )
    assert _pending_of(a) == [("月", "2026-09", None), ("月", "2026-09", "雄輔")]


def test_a_year_waits_until_its_months_are_summarized():
    waiting = _agent(
        {("day_summary", None): ["2025-11", "2025-12"], ("month_summary", None): ["2025-11"]}
    )
    assert ("年", "2025", None) not in _pending_of(waiting)
    ready = _agent(
        {
            ("day_summary", None): ["2025-11", "2025-12"],
            ("month_summary", None): ["2025-11", "2025-12"],
        }
    )
    assert _pending_of(ready) == [("年", "2025", None)]


def test_this_year_is_not_summarized():
    a = _agent({("day_summary", None): ["2026-08"], ("month_summary", None): ["2026-08"]})
    assert _pending_of(a) == []


def test_six_a_night_at_most():
    months = [f"2025-{m:02d}" for m in range(1, 13)]
    a = _agent({("day_summary", None): months})
    r = asyncio.run(rest_calendar.write_calendar(a, now=NOW))
    assert rest_calendar.CALENDAR_MAX_PER_NIGHT == 6
    assert r.written == 6 and r.left == 6
    assert a.backend.complete.await_count == 6


def test_a_month_is_written_at_its_end_with_the_right_direction():
    a = _agent({("day_summary", None): ["2026-08"], ("person_summary", "pid-papa"): ["2026-08"]})
    r = asyncio.run(rest_calendar.write_calendar(a, now=NOW))
    assert r.written == 2
    (fam_mi,), fam_kw = a._oif.write.await_args_list[0].args, a._oif.write.await_args_list[0].kwargs
    assert fam_mi.direction == "月のまとめ"
    assert fam_mi.timestamp == datetime(2026, 8, 31, 23, 59, tzinfo=UTC)
    assert not fam_kw.get("participants")
    (papa_mi,), papa_kw = (
        a._oif.write.await_args_list[1].args,
        a._oif.write.await_args_list[1].kwargs,
    )
    assert papa_mi.direction == "人の月のまとめ"
    assert papa_kw["participants"] == ["pid-papa"]
    # 材料は 1 つ下の段（月なら日ごとの要約）。人ごとはその人の面から
    assert a._oif.tree_children.call_args_list[0].args[0] == "2026-08"
    assert a._oif.tree_children.call_args_list[1].kwargs.get("person_id") == "pid-papa"


def test_the_full_llm_is_asked_with_the_materials():
    a = _agent({("day_summary", None): ["2026-08"]})
    asyncio.run(rest_calendar.write_calendar(a, now=NOW))
    prompt = a.backend.complete.await_args.args[0]
    assert "2026-08 の中身" in prompt and "2026-08" in prompt


def test_a_bad_reply_is_not_written_and_carried_over():
    for reply in ("", "まとめられない", '{"summary": ""}', '{"summary": "' + "あ" * 501 + '"}'):
        a = _agent({("day_summary", None): ["2026-08"]}, reply=reply)
        r = asyncio.run(rest_calendar.write_calendar(a, now=NOW))
        assert r.written == 0 and r.skipped == 1, reply
        a._oif.write.assert_not_awaited()


def test_the_rest_pass_runs_it_after_the_daily_fold():
    from unittest.mock import patch

    from familiar_agent.loop.rest import run_rest_pass
    from familiar_agent.loop.rest_fold import FoldResult

    order: list[str] = []
    agent = MagicMock()
    agent._memory.save_async_with_id = AsyncMock(return_value=("obs1", True))
    agent._observation_perspective = MagicMock(return_value={})

    async def fold(_agent):
        order.append("日次")
        return FoldResult(materials=0, batches=0, written=0, folded=0, skipped=0)

    async def cal(_agent, **_kw):
        order.append("暦")
        return rest_calendar.CalendarResult(written=2, skipped=1, left=5)

    with (
        patch("familiar_agent.loop.rest.fold_since_last_rest", new=fold),
        patch("familiar_agent.loop.rest.write_calendar", new=cal),
    ):
        content = asyncio.run(run_rest_pass(agent))
    assert order == ["日次", "暦"]
    assert "暦のまとめを 2 本書いた（残り 5 本・見送り 1 本）" in content
