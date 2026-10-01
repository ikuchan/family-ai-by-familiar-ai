"""想起の道具 `recall_tree`（記-m 段 2・2026-10-01）。記憶の木の節を指して、その節の要約を引く。

「去年の夏の話」のような遠い過去は、ベクトルの想起では拾いにくい。主LLM が木の節（年 YYYY・月 YYYY-MM・
日 YYYY-MM-DD）を指し、その節の要約だけを読む——年の要約を読んで気になる月を指し、月の要約を読んで
日を指す。節は混ぜて複数指せ、人も指せる（省けば家族全体）。機械が O から引くだけで、LLM は使わない。
書かれていない節は「まだ要約が無い」と返す（主LLM は日を指し直せる）。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

from familiar_agent.loop.event_loop import _FULL_ACTIONS, InformationProcessing
from tests.test_event_loop import _agent as _base_agent, _turn


def _ip(rows_by_node: dict):
    a = _base_agent(stream_returns=[_turn([])])
    calls: list = []

    def tree_summaries(node, *, person_id=None):
        calls.append((node, person_id))
        from familiar_agent.core.memory_tree import parse_node

        if parse_node(node) is None:
            return None
        return rows_by_node.get((node, person_id), [])

    a._oif = MagicMock()
    a._oif.tree_summaries = MagicMock(side_effect=tree_summaries)
    a._pmm.find_person_id_by_name = MagicMock(
        side_effect=lambda name: "pid-papa" if name == "パパ" else None
    )
    return InformationProcessing(a), calls


def _run(ip, tool_input: dict) -> str:
    async def go():
        await ip._run_lookup_body("recall_tree", tool_input, "しらべ", None)
        return ip._triggers.get_nowait().result

    return asyncio.run(go())


def _row(text: str, day: str) -> dict:
    return {"content": text, "timestamp": datetime.fromisoformat(day).replace(tzinfo=timezone.utc)}


def test_the_tool_is_offered_and_says_when_to_use_it():
    assert "recall_tree" in _FULL_ACTIONS
    ip, _ = _ip({})
    defs = {d["name"]: d for d in ip._tools(actions=_FULL_ACTIONS, cache_tools=False)}
    text = defs["recall_tree"]["description"]
    assert "思い出せない" in text and "年" in text and "月" in text


def test_nodes_are_read_each_with_its_heading():
    ip, calls = _ip(
        {
            ("2025", None): [_row("2025 年は引っ越しの年だった", "2025-12-31")],
            ("2025-08", None): [_row("8 月はキャンプに行った", "2025-08-31")],
        }
    )
    out = _run(ip, {"nodes": ["2025", "2025-08"]})
    assert "【2025】" in out and "引っ越しの年" in out
    assert "【2025-08】" in out and "キャンプ" in out
    assert out.index("【2025】") < out.index("【2025-08】")
    assert calls == [("2025", None), ("2025-08", None)]


def test_a_person_is_read_from_their_face():
    ip, calls = _ip({("2025-08-15", "pid-papa"): [_row("パパと花火を見た", "2025-08-15")]})
    out = _run(ip, {"nodes": ["2025-08-15"], "person": "パパ"})
    assert "パパ" in out and "花火" in out
    assert calls == [("2025-08-15", "pid-papa")]


def test_an_unknown_person_is_refused():
    ip, calls = _ip({})
    out = _run(ip, {"nodes": ["2025"], "person": "おばあちゃん"})
    assert "家族に見つからない" in out
    assert calls == []


def test_a_node_without_a_summary_says_so():
    ip, _ = _ip({})
    out = _run(ip, {"nodes": ["2025-09"]})
    assert "【2025-09】" in out and "まだ要約が無い" in out


def test_an_unreadable_node_is_refused():
    ip, _ = _ip({})
    out = _run(ip, {"nodes": ["去年の夏"]})
    assert "読めない" in out and "2025-08" in out  # 書き方を添えて返す


def test_the_body_names_the_tool():
    from familiar_agent.loop.prompt import EVENT_SYSTEM_PROMPT

    line = next(ln for ln in EVENT_SYSTEM_PROMPT.splitlines() if ":id memory" in ln)
    assert "recall_tree" in line
