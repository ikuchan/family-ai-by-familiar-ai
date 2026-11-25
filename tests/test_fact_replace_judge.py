"""記-ah 段 2：夜の REST が、その晩に抜いた事実の置き換えを軽量LLM で判定する（`設計方針_事実の移り変わり` §3）。

候補はコードで出す（その晩に抜いた「答えを隠す向き」の鍵の文に、意味の近い古い記録の鍵）。判定は軽量LLM にまとめて
聞く（本人の決定）。組ごとに両方の記録の本文と時刻を渡し、置き換えたかと、いつからかを返させる。いつからが無ければ
新しい記録の時刻を使う。
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from familiar_agent.core import fact_replace as fr
from familiar_agent.loop import rest_replace


@pytest.fixture(autouse=True)
def _state(monkeypatch):
    """どこまで判定したかの控え（`agent_state`）を辞書に差し替える。"""
    box: dict = {}
    monkeypatch.setattr(rest_replace.state_json, "read", lambda k: box.get(k))
    monkeypatch.setattr(
        rest_replace.state_json, "write", lambda k, v: box.__setitem__(k, v) or True
    )
    return box


T_OLD = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
T_NEW = datetime(2026, 11, 20, 10, 0, tzinfo=timezone.utc)


def _pair(i: int = 0) -> fr.Pair:
    return fr.Pair(
        topic="さぶろうの習い事",
        old_obs=f"old{i}",
        old_key=10 + i,
        old_text="ママ：さぶろうはサッカーに通ってる",
        old_at=T_OLD,
        new_obs=f"new{i}",
        new_key=20 + i,
        new_text="ママ：先月から水泳に変わったよ",
        new_at=T_NEW,
    )


def test_the_prompt_shows_both_records_with_their_dates_and_the_topic():
    p = fr.build_prompt([_pair()])
    assert "さぶろうの習い事" in p
    assert "サッカー" in p and "水泳" in p
    assert "2026-09-01" in p and "2026-11-20" in p
    assert '"since"' in p  # いつからを返させる


def test_a_reply_is_read_into_verdicts_with_dates():
    raw = json.dumps(
        [{"i": 0, "replaced": True, "since": "2026-10-01"}, {"i": 1, "replaced": False}]
    )
    got = fr.parse(raw, 2)
    assert got is not None
    assert got[0] == fr.Verdict(replaced=True, since="2026-10-01")
    assert got[1] == fr.Verdict(replaced=False, since=None)


def test_an_unreadable_reply_is_none_and_unknown_indexes_are_dropped():
    assert fr.parse("考えたけど分からない", 1) is None
    got = fr.parse(json.dumps([{"i": 5, "replaced": True}]), 1)
    assert got == [None]  # 番号が合わない行は捨てる（判定しなかったとみなす）


class _Oif:
    def __init__(self, pairs):
        self.pairs = pairs
        self.written: list[tuple] = []

    def replace_candidates(self, since, *, per_key, min_cos):
        self.args = (since, per_key, min_cos)
        return self.pairs

    def record_replacement(self, old_obs, old_key, new_obs, new_key, *, valid_at):
        self.written.append((old_obs, old_key, new_obs, new_key, valid_at))
        return 1


class _Backend:
    def __init__(self, reply):
        self.reply = reply
        self.prompts: list[str] = []

    async def complete(self, prompt, max_tokens=0):
        self.prompts.append(prompt)
        return self.reply(prompt) if callable(self.reply) else self.reply


def _run(oif, backend, batch=10):
    agent = SimpleNamespace(_oif=oif, _utility_backend=backend, backend=object())
    return asyncio.run(rest_replace.judge(agent, since=T_OLD, batch=batch))


def test_a_replaced_pair_is_written_with_the_date_in_the_words():
    oif = _Oif([_pair()])
    got = _run(oif, _Backend(json.dumps([{"i": 0, "replaced": True, "since": "2026-10-01"}])))
    (old_obs, old_key, new_obs, new_key, at) = oif.written[0]
    assert (old_obs, old_key, new_obs, new_key) == ("old0", 10, "new0", 20)
    assert (at.year, at.month, at.day) == (2026, 10, 1) and at.tzinfo is not None
    assert (got.pairs, got.replaced) == (1, 1)


def test_without_a_date_the_new_record_time_is_used():
    oif = _Oif([_pair()])
    _run(oif, _Backend(json.dumps([{"i": 0, "replaced": True}])))
    assert oif.written[0][4] == T_NEW


def test_a_pair_judged_not_replaced_is_not_written():
    """**反証の側**：置き換えていないと判定した組（同じ話題で両方成り立つ）は書かない。"""
    oif = _Oif([_pair()])
    got = _run(oif, _Backend(json.dumps([{"i": 0, "replaced": False}])))
    assert oif.written == [] and got.replaced == 0


def test_pairs_are_asked_in_batches_and_a_broken_batch_is_counted():
    from dataclasses import replace

    oif = _Oif([replace(_pair(i), new_text=f"ママ：水泳に変わった（{i}）") for i in range(3)])
    replies = iter(["読めない返り", json.dumps([{"i": 0, "replaced": True}])])
    backend = _Backend(lambda _p: next(replies))
    got = _run(oif, backend, batch=2)
    assert len(backend.prompts) == 2  # 2 組と 1 組
    assert got.failed_batches == 1 and got.replaced == 1
    assert oif.written[0][0] == "old2"  # 2 つめの束の 0 番


def test_without_a_light_llm_it_skips():
    agent = SimpleNamespace(_oif=_Oif([_pair()]), _utility_backend=None, backend=None)
    agent._utility_backend = agent.backend = object()
    got = asyncio.run(rest_replace.judge(agent, since=T_OLD))
    assert got.skipped


def test_the_rest_pass_judges_replacements_right_after_extracting_facts():
    """REST の 1 パスは、事実を抜いたすぐ後に置き換えを判定する。判定にかけるのは、抜く前の時刻より後に抜いたもの。"""
    from unittest.mock import AsyncMock, MagicMock, patch

    from familiar_agent.loop import rest_facts as rf
    from familiar_agent.loop.rest import run_rest_pass

    order: list[str] = []
    agent = MagicMock()
    agent._oif.write = AsyncMock(return_value="obs1")
    agent._observation_perspective = MagicMock(return_value={})
    seen: dict = {}

    async def facts(_agent):
        order.append("事実")
        return rf.FactsResult(sources=1, done=1, facts=1, keys=2, failed_batches=0)

    async def judge(_agent, *, since):
        order.append("置き換え")
        seen["since"] = since
        return rest_replace.ReplaceResult(pairs=3, replaced=1, failed_batches=0)

    with (
        patch("familiar_agent.loop.rest.extract_facts_tonight", new=facts),
        patch("familiar_agent.loop.rest.judge_replacements", new=judge),
    ):
        content = asyncio.run(run_rest_pass(agent))
    assert order == ["事実", "置き換え"]
    assert seen["since"].tzinfo is not None
    assert "事実の置き換えを 3 組調べ、1 組を置き換えとした" in content


def test_copies_of_the_same_texts_are_asked_once_and_written_for_each():
    """本文が同じ組（記録の写し）は 1 回だけ聞き、答えをその本文を持つ記録すべてに書く。"""
    from dataclasses import replace

    a = _pair(0)
    b = replace(a, new_obs="new-copy", new_key=99)
    oif = _Oif([a, b])
    backend = _Backend(json.dumps([{"i": 0, "replaced": True}]))
    _run(oif, backend)
    assert len(backend.prompts) == 1 and backend.prompts[0].count("[0]") == 1
    assert {w[2] for w in oif.written} == {"new0", "new-copy"}


def test_a_date_outside_the_two_records_falls_back_to_the_new_record_time():
    """返りのいつからが、古い記録より前か新しい記録より後なら、新しい記録の時刻にする（「来月から」の読み違えなど）。"""
    oif = _Oif([_pair()])
    _run(oif, _Backend(json.dumps([{"i": 0, "replaced": True, "since": "2026-12-01"}])))
    assert oif.written[0][4] == T_NEW


def test_a_failed_night_keeps_the_mark_so_the_next_night_asks_again(_state):
    oif = _Oif([_pair()])
    _run(oif, _Backend("読めない"))
    assert rest_replace.STATE_KEY not in _state  # 進めない
    _run(oif, _Backend(json.dumps([{"i": 0, "replaced": False}])))
    assert rest_replace.STATE_KEY in _state  # 聞き残しが無ければ進める
    # 次の晩は、控えと今晩の始まりの早いほうから拾う
    early = "2026-08-01T00:00:00+00:00"
    _state[rest_replace.STATE_KEY] = early
    _run(oif, _Backend(json.dumps([{"i": 0, "replaced": False}])))
    assert oif.args[0].isoformat() == early
