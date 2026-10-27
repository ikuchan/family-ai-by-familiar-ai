"""最終版どうしを継起で結ぶ（記-t 段 2・2026-10-14・本人の決定）。

継起は、起点どうしでは想起しても意味のない網になる。結ぶのは「聞かれて答えた」の塊（最終版）どうしである。
最終版を書いた直後に背景で、その求めの開始時刻の前 5 分〔仮〕以内に閉じた最終版を候補に引き、Jev が「どれの
続きか」を候補と「どれの続きでもない」から選ぶ。確信度 0.6 未満・失敗・候補なしなら書かない。
"""

from __future__ import annotations

import asyncio
import random
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.core import jev_judges
from familiar_agent.core.final_version import MARK
from familiar_agent.store.relations import KIND_REVISION, KIND_SUCCESSION

# ── 候補の引き方（DB）────────────────────────────────────────────────────────


@pytest.fixture
def store():
    from familiar_agent.db import get_db
    from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID
    from familiar_agent.store.context import StoreContext
    from familiar_agent.store.relations import RelationStore

    db = get_db()
    return RelationStore(
        StoreContext(db=db, lock=db.lock, person_id=DEFAULT_PERSON_ID, embedder=None)
    )


def _put(store, content: str, at: datetime, direction: str = "最終版") -> str:
    oid = str(uuid.uuid4())
    with store._ctx.lock:
        conn = store._ctx.conn()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction) "
                "VALUES (%s, %s, %s, %s)",
                (oid, content, at, direction),
            )
        conn.commit()
    return oid


def _base() -> datetime:
    """ほかの試験と重ならない時刻（試験 DB は共有なので、昔のばらばらな時刻に置く）。"""
    return datetime(1990, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=random.randint(0, 10**7))


def test_only_finals_closed_in_the_last_five_minutes_are_candidates(store):
    start = _base()
    near = _put(store, f"{MARK}近い〕「a」と聞かれ、「b」と答えた", start - timedelta(seconds=60))
    far = _put(store, f"{MARK}遠い〕…", start - timedelta(seconds=400))
    plain = _put(store, "「a」と聞かれ、1番：…", start - timedelta(seconds=30), "求め")
    # 印で始まっても向きが求めなら最終版ではない（記-u 段 1：向きで見分ける）。
    marked = _put(store, f"{MARK}求めの向き〕…", start - timedelta(seconds=30), "求め")
    old = _put(store, f"{MARK}畳まれた〕…", start - timedelta(seconds=30))
    newer = _put(store, f"{MARK}畳んだ側〕…", start - timedelta(seconds=20))
    store.add(KIND_REVISION, [(old, "旧", 0), (newer, "新", 1)])
    me = _put(store, f"{MARK}自分〕…", start - timedelta(seconds=10))

    got = store.recent_finals(
        start - timedelta(seconds=300), start, exclude_id=me, direction="最終版"
    )
    ids = [r["id"] for r in got]
    assert near in ids and newer in ids
    # **反証の側**：遠い・印の無い版・畳まれた・自分自身は入らない。
    for x in (far, plain, marked, old, me):
        assert x not in ids


# ── 判定（偽の Jev）──────────────────────────────────────────────────────────────


def _client(pick, conf=0.9):
    c = MagicMock()
    c.available = True
    c.ask = AsyncMock(
        return_value=JevAnswer(ok=True, answers={"follows": {"choice": pick, "confidence": conf}})
    )
    return c


_CANDS = {"c1": f"{MARK}10:00〜10:01〕「天気は？」と聞かれ、「晴れ」と答えた"}


def _judge(client):
    return asyncio.run(
        jev_judges.judge_succession(
            client,
            current=f"{MARK}10:02〜10:03〕「じゃあ傘は？」と聞かれ、「要らない」と答えた",
            candidates=_CANDS,
            min_conf=0.6,
        )
    )


def test_the_chosen_candidate_is_returned():
    c = _client("c1")
    assert _judge(c) == "c1"
    _state, questions = c.ask.await_args.args
    assert set(questions["follows"]["criteria"]) == {"c1", "none"}


def test_none_low_confidence_and_failure_link_nothing():
    assert _judge(_client("none")) is None
    assert _judge(_client("c1", 0.5)) is None
    failing = MagicMock(available=True)
    failing.ask = AsyncMock(return_value=JevAnswer(ok=False, error="時間切れ"))
    assert _judge(failing) is None
    assert _judge(None) is None


def test_a_long_state_shrinks_each_candidate_but_keeps_them_all():
    from familiar_agent.core.final_version import fit

    cands = {f"c{i}": "あ" * 8000 for i in range(5)}
    cur, got = fit("い" * 8000, cands, 28_000)
    assert set(got) == set(cands)  # 候補の数は減らさない（字数だけを縮める）
    assert len(cur) + sum(len(v) for v in got.values()) <= 28_000
    assert len(cur) < 8000 and all(len(v) < 8000 for v in got.values())
    # **反証の側**：収まっていれば切らない。
    short = {"c1": "う" * 100}
    assert fit("え" * 100, short, 28_000) == ("え" * 100, short)


# ── 結ぶ（ループ）─────────────────────────────────────────────────────────────


def _ip():
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    a.config.completion_content_max = 8192
    a.config.jev_confidence_min = 0.6
    a.config.jev_state_max_chars = 28_000
    a.config.memory.succession_window_sec = 300
    ip = InformationProcessing(a)
    ip._req.request_id = "origin-1"
    ip._req.request_text = "じゃあ傘は？"
    ip._req.trigger_kind = "発話"
    ip._req.began_at = datetime.now(timezone.utc)
    return a, ip


def _close(ip, text="要らない"):
    async def go():
        await ip._finish(text, [], "発話")
        await ip.settle_finals()
        await ip.close()

    asyncio.run(go())


def _successions(a):
    return [c.args[1] for c in a._memory.link.call_args_list if c.args[0] == KIND_SUCCESSION]


def test_closing_links_the_final_to_the_chosen_earlier_final():
    a, ip = _ip()
    a._memory.recent_finals = MagicMock(
        return_value=[{"id": "prev-final", "content": _CANDS["c1"], "timestamp": None}]
    )
    a._jev = _client("prev-final")
    _close(ip)
    links = _successions(a)
    assert len(links) == 1
    prev, cur = links[0]
    assert prev == ("prev-final", "前", 0)
    assert cur[1:] == ("後", 1) and cur[0].startswith("obs")
    # 候補は「開始時刻の前 300 秒から開始時刻まで」で引く。
    since, until = a._memory.recent_finals.call_args.args[:2]
    assert until - since == timedelta(seconds=300)


def test_no_candidates_means_no_question():
    """**反証の側**：候補が無ければ Jev を呼ばない。"""
    a, ip = _ip()
    a._memory.recent_finals = MagicMock(return_value=[])
    a._jev = _client("prev-final")
    _close(ip)
    a._jev.ask.assert_not_awaited()
    assert _successions(a) == []


def test_not_a_continuation_links_nothing():
    a, ip = _ip()
    a._memory.recent_finals = MagicMock(
        return_value=[{"id": "prev-final", "content": _CANDS["c1"], "timestamp": None}]
    )
    a._jev = _client("none")
    _close(ip)
    assert _successions(a) == []


def test_a_summarized_final_is_linked_too():
    """背景で要約して書いた最終版も、書いた直後に結ぶ。"""
    from familiar_agent.loop.request import Lookup

    a, ip = _ip()
    a.config.completion_content_max = 300
    a.backend.complete = AsyncMock(return_value="傘は要らないと答えた")
    ip._req.lookups = [
        Lookup(index=1, action="search_deferred", query="雨", generation=0, result="長い" * 300)
    ]
    a._memory.recent_finals = MagicMock(
        return_value=[{"id": "prev-final", "content": _CANDS["c1"], "timestamp": None}]
    )
    a._jev = _client("prev-final")
    _close(ip)
    assert len(_successions(a)) == 1
