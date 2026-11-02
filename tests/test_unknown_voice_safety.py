"""知らない声で SAFETY が溜まり、見回して書いた文を声にしない（情-r・2026-11-02・本人の決定ア・イ）。

11/02 22:26〜22:28（再起動の直後）、知らない声のたびに SAFETY を +0.10 し、1 分ほどで 1.00 に届いて 2 回発火した。知らない声と
判定された 33 回は、いちばん近い家族の声との似かたが 0.01〜0.33（中央 0.17）で、家族と認める基準（在席あり 0.35）に届かなかった。
発火すると調停が見回しを 5 回続け、上限の回は調停が見回しを選んでも主LLM へ回り、画像つきで文を書かせたが、SAFETY の発火なので
声にしなかった（10/14 に記録したのと同じ流れ）。

- ア：顔ぶれに、誰か分かった人（顔・声・手入力のどれかで・`get_present_ids`・古い顔ぶれは除く）が居れば、知らない声で SAFETY を
  上げない（名前の分からない札だけ・在席だけなら、いまどおり上げる）。顔ぶれの記録は顔・声・手入力を区別しないので、「顔で」でなく
  「誰か分かった人」と読み直した（本人の決定ア）。
- イ：SAFETY の発火の求めは、見回しを反復の上限まで続けてよいが、上限の回は主LLM を呼ばずに黙って閉じる。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.test_voice_nudges_bond_or_safety import KNOWN, STRANGER, _drop, _ip, _nudged

pytestmark = pytest.mark.real_window


# ── ア：誰か分かった人が顔ぶれに居れば上げない ─────────────────────────────────────


def test_a_stranger_voice_with_a_known_person_in_view_does_not_nudge_safety():
    ip, a = _ip(here=True)
    a._pmm.get_present_ids = MagicMock(return_value=["person-1"])
    assert _drop(ip, voice=STRANGER) is True
    assert _nudged(a) == []


def test_only_an_unnamed_face_still_nudges_safety():
    """**反証の側**：見えているのが名前の分からない札だけ（在席だけ）なら、いまどおり上げる。"""
    ip, a = _ip(here=True)
    a._pmm.get_present_ids = MagicMock(return_value=[])
    assert _drop(ip, voice=STRANGER) is True
    assert _nudged(a) == [("safety", pytest.approx(0.10))]


def test_a_known_voice_still_nudges_bond():
    """**反証の側**：知っている声は、顔が見えていてもいまどおり BOND。"""
    ip, a = _ip(here=True)
    a._pmm.get_present_ids = MagicMock(return_value=["person-1"])
    assert _drop(ip, voice=KNOWN) is True
    assert _nudged(a) == [("bond", pytest.approx(0.05))]


# ── イ：SAFETY の発火は、上限の回に主LLM を呼ばずに閉じる ──────────────────────────


def _capped(trigger_kind: str, axis: str):
    from familiar_agent.loop.arbiter import Decision
    from familiar_agent.loop.iteration import Iteration

    ip = MagicMock()
    ip._agent.config.event_max_iterations = 5
    ip._request_generation = 1
    ip._req.trigger_kind = trigger_kind
    ip._req.fired_axis = axis
    ip._finish = AsyncMock()
    # 判定は本物（ループの `_quiet_at_cap`）を使う
    from functools import partial

    from familiar_agent.loop.event_loop import InformationProcessing

    ip._quiet_at_cap = partial(InformationProcessing._quiet_at_cap, ip)
    it = Iteration(ip)
    it.gen = 1
    it.capped = True
    it.decision = Decision(branch="action", action="look", query="見回す")
    it.ws = SimpleNamespace(memories=[], returned_actions=frozenset({"look"}), for_main="")
    return ip, asyncio.run(it._close_early())


def test_a_safety_firing_at_the_cap_closes_without_the_main_llm():
    ip, out = _capped("情動", "safety")
    assert out == ""
    ip._finish.assert_awaited_once()
    assert ip._finish.await_args.args[2] == "沈黙"


@pytest.mark.parametrize(("kind", "axis"), [("情動", "bond"), ("発話", "")])
def test_other_requests_at_the_cap_still_go_to_the_main_llm(kind, axis):
    """**反証の側**：SAFETY 以外の発火や会話の求めは、いまどおり主LLM へ回る（閉じない）。"""
    ip, out = _capped(kind, axis)
    assert out is None
    ip._finish.assert_not_awaited()


def test_a_stale_known_face_does_not_count():
    """**反証の側**：しばらく合図の無い（古い）顔ぶれは、もう見えていないとみて SAFETY を上げる。"""
    ip, a = _ip(here=True)
    a._pmm.get_present_ids = MagicMock(return_value=["person-1"])
    a._pmm.stale_present_ids = MagicMock(return_value=["person-1"])
    assert _drop(ip, voice=STRANGER) is True
    assert _nudged(a) == [("safety", pytest.approx(0.10))]
