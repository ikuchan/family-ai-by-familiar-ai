"""共起は、そのターンの起点と、Jev が答えに使ったと申告した記憶だけで作る（記-s・2026-10-14・本人の決定）。

以前は、ターン後にそのターンの W の全部と、そのターンに作った記録を 1 本の共起に結んでいた（`_record_cooccurrence`）。
1 本が十数件を結ぶので、その日の共起と継起をたどると 1 日が 400〜650 件の 1 つの塊になった（出-bq の計測）。

- 項は、起点 1 件（判定しない）と、申告が important・referred・useless の記憶（unused は入れない）。
- 使った記憶が 2 件未満なら書かない（2 つにしかかからない網は作らない）。
- 書くのは申告を当てる 1 か所（`workspace.apply_memory_verdicts`）。背景の申告は、投げた時点の起点を使う。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop import workspace
from familiar_agent.store.relations import KIND_COOCCURRENCE, cooccurring_members

_MAP = {"aaaaaaaaaaaa": "a-full", "bbbbbbbbbbbb": "b-full", "cccccccccccc": "c-full"}


def _v(**kw):
    return [{"id": k, "verdict": v} for k, v in kw.items()]


# ── 項と条件（純関数）─────────────────────────────────────────────────────────


def test_the_origin_and_the_used_memories_are_the_members():
    got = cooccurring_members(
        "origin", {"a-full": "important", "b-full": "useless", "c-full": "unused"}
    )
    assert got == ["origin", "a-full", "b-full"]


def test_one_used_memory_is_not_enough():
    """**反証の側**：起点と記憶 1 件の 2 つにしかかからない網は作らない。"""
    assert cooccurring_members("origin", {"a-full": "referred", "b-full": "unused"}) is None
    assert cooccurring_members("origin", {}) is None


def test_no_origin_writes_nothing():
    assert cooccurring_members("", {"a-full": "referred", "b-full": "referred"}) is None


def test_the_origin_is_not_doubled():
    got = cooccurring_members(
        "a-full", {"a-full": "referred", "b-full": "referred", "c-full": "referred"}
    )
    assert got == ["a-full", "b-full", "c-full"]


# ── 申告を当てる所で書く ─────────────────────────────────────────────────────────


def test_applying_verdicts_writes_one_cooccurrence():
    oif = MagicMock()
    workspace.apply_memory_verdicts(
        oif,
        "__self__",
        _v(aaaaaaaaaaaa="referred", bbbbbbbbbbbb="important", cccccccccccc="unused"),
        _MAP,
        origin="origin",
    )
    oif.link.assert_called_once_with(
        KIND_COOCCURRENCE, [("origin", "項", None), ("a-full", "項", None), ("b-full", "項", None)]
    )


def test_applying_too_few_verdicts_writes_none():
    oif = MagicMock()
    workspace.apply_memory_verdicts(
        oif, "__self__", _v(aaaaaaaaaaaa="referred", bbbbbbbbbbbb="unused"), _MAP, origin="origin"
    )
    oif.link.assert_not_called()


def test_without_an_origin_nothing_is_written():
    oif = MagicMock()
    workspace.apply_memory_verdicts(
        oif, "__self__", _v(aaaaaaaaaaaa="referred", bbbbbbbbbbbb="referred"), _MAP
    )
    oif.link.assert_not_called()


# ── 背景の申告は、投げた時点の起点を使う ─────────────────────────────────────────────


def test_the_background_declaration_keeps_the_origin_it_was_thrown_with(monkeypatch):
    from tests.test_unsaid_is_recorded_and_folded import _ip

    a, ip = _ip()
    a._oif = MagicMock()
    ip._req.request_id = "origin-1"
    gate = asyncio.Event()

    async def judged(*_a, **_k):
        await gate.wait()
        return _v(aaaaaaaaaaaa="referred", bbbbbbbbbbbb="referred")

    monkeypatch.setattr("familiar_agent.core.jev_judges.judge_verdicts", judged)

    async def go():
        ip._declare_light_memory_use(
            utterance="おはよう",
            reply="おはよう",
            workspace_ctx="",
            w_id_map=_MAP,
            verdict_view="__self__",
            memories=[],
            spoken=True,
        )
        ip._req.request_id = "origin-2"  # 届くまでに次の求めが始まった
        gate.set()
        await asyncio.gather(*list(ip._verdict_tasks))

    asyncio.run(go())
    members = a._oif.link.call_args.args[1]
    assert members[0] == ("origin-1", "項", None)


# ── 主LLM の道 ─────────────────────────────────────────────────────────────────


def test_a_checked_main_reply_writes_with_its_origin():
    from tests.test_send_back_is_dispatched import _ip, _say
    from tests.test_verdicts_come_from_jev_only import _oif, _run_with_map

    ip, a = _ip()
    _oif(a)
    ip._req.request_id = "origin-main"
    ip._speech_check_violation = AsyncMock(
        return_value=(None, _v(abcdef123456="referred", fedcba654321="useless"))
    )
    _run_with_map(ip, _say("はい"), w_map={"abcdef123456": "m1", "fedcba654321": "m2"})
    a._oif.link.assert_any_call(
        KIND_COOCCURRENCE,
        [("origin-main", "項", None), ("m1", "項", None), ("m2", "項", None)],
    )


def test_a_sent_back_main_reply_writes_none():
    from tests.test_send_back_is_dispatched import _ip, _say
    from tests.test_verdicts_come_from_jev_only import _oif, _run_with_map

    ip, a = _ip()
    _oif(a)
    ip._req.request_id = "origin-main"
    ip._speech_check_violation = AsyncMock(
        return_value=("決まり x：y", _v(abcdef123456="referred", fedcba654321="useless"))
    )
    _run_with_map(ip, _say("はい"), w_map={"abcdef123456": "m1", "fedcba654321": "m2"})
    kinds = [c.args[0] for c in a._oif.link.call_args_list]
    assert KIND_COOCCURRENCE not in kinds


# ── ターン後の処理は、もう共起を書かない ───────────────────────────────────────────


def test_the_post_response_pipeline_no_longer_writes_cooccurrence():
    """**反証の側**：古い書き手が残っていない。"""
    import inspect

    from familiar_agent.agent import EmbodiedAgent

    assert not hasattr(EmbodiedAgent, "_record_cooccurrence")
    params = inspect.signature(EmbodiedAgent._run_post_response_pipeline).parameters
    assert "extra_cooccurring_ids" not in params
