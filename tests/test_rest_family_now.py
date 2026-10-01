"""REST が家族のいまの様子を書き直す（知-ad 段 3・2026-10-01・本人の決定）。

材料は日々の蒸留（日次の畳み込み）が書いた**関係のまとめ（`person_summary`）だけ**。家族ひとりずつ、まだ読んで
いない関係のまとめを**古い順に 10 本ずつ**読み、そのたびに前の版へ重ねて書き直す（10 本＝その人と関わった
10 日ぶん）。読み終えた位置は最後に読んだ関係のまとめの時刻として控え、次はその先から読む。1 晩のうちに 10 本
そろうかぎり繰り返し、届かない残りは次の晩へ。書き手はフルLLM、300 字以内。返りが空・読めない・超過なら
書かず、位置も進めない（次の晩にもう一度試す）。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.core import family_now as fn
from familiar_agent.loop import rest_family_now as rfn

UTC = timezone.utc
FAMILY = """## パパ

- **名前**：雄輔
- **呼び方**：パパ、ゆうすけ
- **性格・傾向**：話好き

## たいき

- **名前**：泰輝
- **呼び方**：たいき
"""


def _notes(n: int, start: datetime) -> list[dict]:
    return [
        {"content": f"その日のこと {i}", "timestamp": start + timedelta(days=i)} for i in range(n)
    ]


@pytest.fixture
def state(monkeypatch):
    """DB の代わりに辞書で持つ（`family_now.stored`・`update`）。"""
    data: dict = {}

    def update(name, text, *, counted_from):
        old = data.get(name)
        data[name] = fn.Now(text=text, before=old.text if old else "", counted_from=counted_from)
        return True

    monkeypatch.setattr(fn, "stored", lambda: dict(data))
    monkeypatch.setattr(fn, "update", update)
    return data


def _agent(notes_by_pid: dict, *, replies=None):
    a = MagicMock()
    a._family_md = FAMILY
    a._pmm.find_person_id_by_name = MagicMock(
        side_effect=lambda name: {"雄輔": "pid-papa", "泰輝": "pid-taiki"}.get(name)
    )

    def after(pid, since):
        return [r for r in notes_by_pid.get(pid, []) if r["timestamp"] > since]

    a._oif.person_notes_after = MagicMock(side_effect=after)
    texts = iter(replies or [f'{{"now": "様子 {i}"}}' for i in range(1, 100)])
    a.backend.complete = AsyncMock(side_effect=lambda *_a, **_k: next(texts))
    return a


T0 = datetime(2026, 9, 14, 23, 59, tzinfo=UTC)


def test_ten_new_notes_make_one_rewrite_from_the_oldest(state):
    a = _agent({"pid-papa": _notes(13, T0)})
    r = asyncio.run(rfn.update_family_now(a))
    assert r.rewrites == {"パパ": 1}
    assert state["パパ"].text == "様子 1"
    assert state["パパ"].counted_from == T0 + timedelta(
        days=9
    )  # 10 本目の時刻。残り 3 本は次の晩へ
    prompt = a.backend.complete.await_args.args[0]
    assert "その日のこと 0" in prompt and "その日のこと 9" in prompt
    assert "その日のこと 10" not in prompt


def test_nine_notes_are_not_enough(state):
    a = _agent({"pid-papa": _notes(9, T0)})
    r = asyncio.run(rfn.update_family_now(a))
    assert r.rewrites == {}
    a.backend.complete.assert_not_awaited()


def test_a_backlog_is_read_ten_at_a_time_as_far_as_it_goes(state):
    a = _agent({"pid-papa": _notes(25, T0)})
    r = asyncio.run(rfn.update_family_now(a))
    assert r.rewrites == {"パパ": 2}
    assert state["パパ"].text == "様子 2" and state["パパ"].before == "様子 1"
    second = a.backend.complete.await_args_list[1].args[0]
    assert "様子 1" in second  # 前の版に重ねる
    assert "その日のこと 10" in second and "その日のこと 19" in second


def test_the_next_night_reads_after_where_it_stopped(state):
    state["パパ"] = fn.Now(text="前の様子", before="", counted_from=T0 + timedelta(days=9))
    a = _agent({"pid-papa": _notes(20, T0)})
    asyncio.run(rfn.update_family_now(a))
    prompt = a.backend.complete.await_args.args[0]
    assert "前の様子" in prompt and "その日のこと 10" in prompt and "その日のこと 9\n" not in prompt


def test_the_persons_own_section_is_given(state):
    a = _agent({"pid-papa": _notes(10, T0)})
    asyncio.run(rfn.update_family_now(a))
    prompt = a.backend.complete.await_args.args[0]
    assert "話好き" in prompt  # 人が書いた `FAMILY.md` のその人の節
    assert "泰輝" not in prompt  # ほかの人の節は渡さない


@pytest.mark.parametrize("reply", ["", "書けない", '{"now": ""}', '{"now": "' + "あ" * 301 + '"}'])
def test_a_bad_reply_keeps_the_old_and_does_not_move(state, reply):
    a = _agent({"pid-papa": _notes(20, T0)}, replies=[reply, '{"now": "使われない"}'])
    r = asyncio.run(rfn.update_family_now(a))
    assert r.rewrites == {} and r.skipped == 1
    assert "パパ" not in state
    assert a.backend.complete.await_count == 1  # 落ちたらその人はその晩やめる


def test_each_person_is_read_separately(state):
    a = _agent({"pid-papa": _notes(10, T0), "pid-taiki": _notes(10, T0)})
    r = asyncio.run(rfn.update_family_now(a))
    assert r.rewrites == {"パパ": 1, "たいき": 1}


def test_the_rest_pass_runs_it_after_the_calendar():
    from unittest.mock import patch

    from familiar_agent.loop import rest_calendar
    from familiar_agent.loop.rest import run_rest_pass
    from familiar_agent.loop.rest_fold import FoldResult

    order: list[str] = []
    agent = MagicMock()
    agent._memory.save_async_with_id = AsyncMock(return_value=("obs1", True))
    agent._observation_perspective = MagicMock(return_value={})

    async def fold(_agent):
        return FoldResult(materials=0, batches=0, written=0, folded=0, skipped=0)

    async def cal(_agent, **_kw):
        order.append("暦")
        return rest_calendar.CalendarResult()

    async def now(_agent):
        order.append("いまの様子")
        return rfn.FamilyNowResult(rewrites={"パパ": 2})

    with (
        patch("familiar_agent.loop.rest.fold_since_last_rest", new=fold),
        patch("familiar_agent.loop.rest.write_calendar", new=cal),
        patch("familiar_agent.loop.rest.update_family_now", new=now),
    ):
        content = asyncio.run(run_rest_pass(agent))
    assert order == ["暦", "いまの様子"]
    assert "いまの様子を書き直した（パパ 2 回）" in content
