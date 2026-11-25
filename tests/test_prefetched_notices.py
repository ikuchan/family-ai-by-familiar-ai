"""記-ag 段 2・段 3：先回りして調べた結果を、知らせ待ちに置き、人がそばに居れば知らせる（`設計方針_先回りして調べておく`）。

段 2：SEEKING の求めで調べものの結果が届いて閉じたら、軽量LLM に「家族に知らせる値打ちがあるか」と、知らせる中身と、
何時間で古くなるかを聞き、値打ちがあれば知らせ待ち（`agent_state`）に置く。
段 3：T（自律機構）の輪が、そばに人が居て黙っていなければ、知らせ待ちのうち古くなっていないものを取り出し、`機器` の
求め「調べておいたこと」を積む（主LLM が話す・在席の門はいつものまま）。古くなったものは知らせずに捨てる。
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from familiar_agent.config import DriveConfig
from familiar_agent.loop import prefetch

NOW = datetime(2026, 11, 25, 9, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _box(monkeypatch):
    """知らせ待ち（`agent_state`）を辞書に差し替える。"""
    box: dict = {}
    monkeypatch.setattr(prefetch.state_json, "read_checked", lambda k: (True, box.get(k)))
    monkeypatch.setattr(prefetch.state_json, "write", lambda k, v: box.__setitem__(k, v) or True)
    return box


class _Backend:
    def __init__(self, reply):
        self.reply = reply
        self.prompts: list[str] = []

    async def complete(self, prompt, max_tokens=0):
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _agent(reply):
    return SimpleNamespace(_utility_backend=_Backend(reply), backend=object())


def _keep(agent, results="- 昨日の試合の結果：2対1で勝った"):
    return asyncio.run(prefetch.keep(agent, results, now=NOW))


def test_a_result_worth_telling_is_kept_until_it_gets_old(_box):
    agent = _agent(json.dumps({"tell": True, "what": "昨日の試合は 2 対 1 で勝った", "hours": 6}))
    assert _keep(agent) is True
    assert "昨日の試合の結果" in agent._utility_backend.prompts[0]  # 結果を見せて聞く
    (item,) = _box[prefetch.STATE_KEY]
    assert item["what"] == "昨日の試合は 2 対 1 で勝った"
    assert datetime.fromisoformat(item["until"]) == NOW + timedelta(hours=6)


def test_a_result_not_worth_telling_is_not_kept(_box):
    """**反証の側**：値打ちが無いと判定したもの、読めない返り、呼べなかったときは置かない。"""
    assert _keep(_agent(json.dumps({"tell": False}))) is False
    assert _keep(_agent("よく分からない")) is False
    assert _keep(_agent(RuntimeError("落ちた"))) is False
    assert prefetch.STATE_KEY not in _box


def test_without_a_light_llm_nothing_is_asked(_box):
    agent = SimpleNamespace()
    agent._utility_backend = agent.backend = _Backend(json.dumps({"tell": True, "what": "x"}))
    assert _keep(agent) is False and agent.backend.prompts == []


def test_hours_are_clamped_and_default_when_missing(_box):
    _keep(_agent(json.dumps({"tell": True, "what": "a", "hours": 1000})))
    _keep(_agent(json.dumps({"tell": True, "what": "b"})))
    a, b = _box[prefetch.STATE_KEY]
    assert datetime.fromisoformat(a["until"]) == NOW + timedelta(
        hours=DriveConfig().prefetch_max_hours
    )
    assert datetime.fromisoformat(b["until"]) == NOW + timedelta(
        hours=DriveConfig().prefetch_default_hours
    )


def test_due_notices_are_told_once_and_old_ones_are_dropped(_box):
    _box[prefetch.STATE_KEY] = [
        {
            "what": "古い天気",
            "at": NOW.isoformat(),
            "until": (NOW - timedelta(hours=1)).isoformat(),
        },
        {
            "what": "試合の結果",
            "at": NOW.isoformat(),
            "until": (NOW + timedelta(hours=3)).isoformat(),
        },
    ]
    assert prefetch.take_due(now=NOW) == ["試合の結果"]
    assert prefetch.take_due(now=NOW + timedelta(hours=1)) == []  # 知らせ済みは繰り返さない
    (left,) = _box[prefetch.STATE_KEY]  # 古いものは捨てた
    assert (
        left["what"] == "試合の結果" and left["told"] is True
    )  # 知らせ済みも古くなるまで残す（判定に見せる）


def test_the_same_news_is_not_kept_twice_and_known_ones_are_shown_to_the_judge(_box):
    reply = json.dumps({"tell": True, "what": "昨日の試合は 2 対 1 で勝った"})
    assert _keep(_agent(reply)) is True
    again = _agent(reply)
    assert _keep(again) is False  # 同じ中身は置かない
    assert "昨日の試合は 2 対 1 で勝った" in again._utility_backend.prompts[0]  # 判定に見せる
    assert len(_box[prefetch.STATE_KEY]) == 1 and not _box[prefetch.STATE_KEY][0].get("told")


def test_the_waiting_notices_are_read_at_most_every_interval(_box, monkeypatch):
    """輪は 0.5 秒ごとに回るが、知らせ待ちの DB は `prefetch_check_sec` に 1 回しか読まない。"""
    reads: list = []
    monkeypatch.setattr(prefetch, "take_due", lambda now: reads.append(now) or [])
    t, _ = _tonic(here=True)
    asyncio.run(t._maybe_tell_prefetched())
    asyncio.run(t._maybe_tell_prefetched())  # すぐ次の周は読まない
    assert len(reads) == 1
    t._prefetch_checked_at -= t._cfg.prefetch_check_sec + 1
    asyncio.run(t._maybe_tell_prefetched())
    assert len(reads) == 2


def test_a_failed_write_offers_nothing(_box, monkeypatch):
    """**反証の側**：差し出した時刻を書けなければ何も返さない（書けないまま毎回積まない）。"""
    _box[prefetch.STATE_KEY] = [
        {
            "what": "試合の結果",
            "at": NOW.isoformat(),
            "until": (NOW + timedelta(hours=3)).isoformat(),
        }
    ]
    monkeypatch.setattr(prefetch.state_json, "write", lambda k, v: False)
    assert prefetch.take_due(now=NOW) == []


def test_an_unreadable_time_is_dropped_without_breaking_the_rest(_box):
    _box[prefetch.STATE_KEY] = [
        {"what": "時差の無い時刻", "until": "2026-11-25T21:00:00"},
        {"what": "試合の結果", "until": (NOW + timedelta(hours=3)).isoformat()},
    ]
    assert prefetch.take_due(now=NOW) == ["試合の結果"]


def test_the_content_asks_to_tell_what_was_looked_up():
    content = prefetch.content(["試合の結果", "明日の天気"])
    assert "試合の結果" in content and "明日の天気" in content
    assert "知らせる" in content


# ── 段 3：T の輪が知らせる ──────────────────────────────────────────


def _tonic(*, here: bool, silenced: bool = False, sleeping: bool = False, music: bool = False):
    from familiar_agent.loop.tonic import Tonic

    pushed: list = []
    t = Tonic.__new__(Tonic)
    t._cfg = DriveConfig()
    t._someone_here = lambda: here  # type: ignore[method-assign]
    t._agent = SimpleNamespace(
        _music_state=SimpleNamespace(playing=True) if music else None,
        paju_state=SimpleNamespace(
            silenced_outside=lambda through_silence=False: silenced, is_sleeping=lambda: sleeping
        ),
    )
    t._dif = SimpleNamespace(
        device=lambda kind, content, passes_gate=False: pushed.append((kind, content, passes_gate))
    )
    return t, pushed


def _pending(_box):
    _box[prefetch.STATE_KEY] = [
        {"what": "試合の結果", "at": NOW.isoformat(), "until": "2999-01-01T00:00:00+00:00"}
    ]


def test_someone_here_gets_a_device_request_to_tell(_box):
    _pending(_box)
    t, pushed = _tonic(here=True)
    asyncio.run(t._maybe_tell_prefetched())
    ((kind, content, passes_gate),) = pushed
    assert kind == prefetch.KIND and "試合の結果" in content
    assert passes_gate is False  # 在席の門はいつものまま


def test_nobody_here_or_silenced_keeps_waiting(_box):
    """**反証の側**：誰も居ない・黙っていて・寝ている・音楽が鳴っているあいだは積まず、知らせ待ちに残す。"""
    _pending(_box)
    for t, pushed in (
        _tonic(here=False),
        _tonic(here=True, silenced=True),
        _tonic(here=True, sleeping=True),
        _tonic(here=True, music=True),
    ):
        asyncio.run(t._maybe_tell_prefetched())
        assert pushed == []
    assert len(_box[prefetch.STATE_KEY]) == 1


# ── 段 2：SEEKING の求めが閉じたら判定を裏で回す ────────────────────


def _loop(*, trigger_kind: str, axis: str, results: str):
    from familiar_agent.loop.event_loop import InformationProcessing

    ip = InformationProcessing.__new__(InformationProcessing)
    ip._req = SimpleNamespace(trigger_kind=trigger_kind, fired_axis=axis)
    ip._search_results = lambda: results  # type: ignore[method-assign]
    ip._verdict_tasks = set()
    ip._agent = object()
    return ip


def test_a_closing_seeking_request_with_results_is_judged(monkeypatch):
    seen: list = []

    async def keep(agent, results, *, now):
        seen.append(results)
        return True

    monkeypatch.setattr(prefetch, "keep", keep)

    async def run():
        ip = _loop(trigger_kind="情動", axis="seeking", results="- 試合：勝った")
        ip._keep_prefetched()
        await asyncio.gather(*ip._verdict_tasks)

    asyncio.run(run())
    assert seen == ["- 試合：勝った"]


def test_other_requests_and_empty_results_are_not_judged(monkeypatch):
    """**反証の側**：人の言葉への求め、ほかの軸の発火、結果の無い SEEKING は判定しない。"""

    async def keep(*a, **k):
        raise AssertionError("呼ばない")

    monkeypatch.setattr(prefetch, "keep", keep)

    async def run():
        for kind, axis, res in (
            ("発話", "", "- x"),
            ("情動", "safety", "- x"),
            ("情動", "seeking", ""),
        ):
            ip = _loop(trigger_kind=kind, axis=axis, results=res)
            ip._keep_prefetched()
            assert ip._verdict_tasks == set()

    asyncio.run(run())


def test_at_most_a_few_are_offered_in_one_turn(_box):
    _box[prefetch.STATE_KEY] = [
        {"what": f"知らせ{i}", "until": "2999-01-01T00:00:00+00:00"} for i in range(5)
    ]
    assert len(prefetch.take_due(now=NOW)) == prefetch.PER_TURN


def test_an_unreadable_store_is_not_overwritten(_box, monkeypatch):
    """**反証の側**：知らせ待ちを読めないときは、判定も書き込みもしない（空として上書きして消さない）。"""
    _box[prefetch.STATE_KEY] = [{"what": "試合の結果", "until": "2999-01-01T00:00:00+00:00"}]
    monkeypatch.setattr(prefetch.state_json, "read_checked", lambda k: (False, None))
    agent = _agent(json.dumps({"tell": True, "what": "天気"}))
    assert _keep(agent) is False and agent._utility_backend.prompts == []
    assert prefetch.take_due(now=NOW) == []
    assert len(_box[prefetch.STATE_KEY]) == 1


def test_a_failing_state_check_does_not_tell(_box):
    """黙っているか寝ているかを確かめられないときは、知らせない側に倒す。"""
    _pending(_box)
    t, pushed = _tonic(here=True)

    def broken(through_silence=False):
        raise RuntimeError("移り途中")

    t._agent.paju_state.silenced_outside = broken
    asyncio.run(t._maybe_tell_prefetched())
    assert pushed == []


def test_a_nan_hours_falls_back_to_the_default(_box):
    _keep(_agent('{"tell": true, "what": "a", "hours": NaN}'))
    (item,) = _box[prefetch.STATE_KEY]
    hours = DriveConfig().prefetch_default_hours
    assert datetime.fromisoformat(item["until"]) == NOW + timedelta(hours=hours)
