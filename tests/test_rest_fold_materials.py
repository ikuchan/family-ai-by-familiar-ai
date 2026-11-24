"""日次の畳み込みの材料を引く口（記-a-ろ-は・2026-09-14）。

前回の内省（`direction='内省'` の最新）より後の記録のうち、指定の `direction` で、まだ畳まれて
おらず（役割 `旧` が無い）、核でない（根づき $n < 1$）ものを古い順に返す。`内省` が無ければ
全期間。`求め`（版）・`保留`・`内省`・`記憶` は呼び手が `direction` で外す。
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch


from familiar_agent.person_memory_manager import AGENT_SELF_ID

_DB_URL = os.environ["DATABASE_URL"]
_VEC = "[" + ",".join(["1"] + ["0"] * 1023) + "]"
FOLD_DIRECTIONS = ("発話", "会話", "観察", "独白", "情動", "機器")


from tests._db import autocommit_conn as _conn  # noqa: E402


def _plant(cur, content: str, ts: datetime, *, direction="発話", n=0, hidden=False) -> str:
    oid = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO observations (id, content, timestamp, direction, emotion, groundedness_g0)"
        " VALUES (%s, %s, %s, %s, 'neutral', 0.5)",
        (oid, content, ts, direction),
    )
    cur.execute(
        "INSERT INTO situated_memories (id, obs_id, person_id, vector, relation_key, groundedness_n)"
        " VALUES (%s, %s, %s, %s, 'present', %s)",
        (str(uuid.uuid4()), oid, AGENT_SELF_ID, _VEC, n),
    )
    if hidden:
        cur.execute("INSERT INTO relations (kind) VALUES ('畳み込み') RETURNING id")
        rid = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO relation_members (relation_id, obs_id, role, position) VALUES (%s, %s, '旧', 0)",
            (rid, oid),
        )
    return oid


def _oif():
    from familiar_agent.io.oif import OIF
    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    with patch.object(_EmbeddingModel, "pre_warm"):
        return OIF(ObservationMemory().for_person(AGENT_SELF_ID))


def test_materials_are_the_uncore_unfolded_records_regardless_of_the_last_rest():
    """前回の内省の時刻には依らない（記-l・2026-09-15）。

    初回の REST は 2,677 件を 69 回に分けて頼み 14 分かかった。上限で翌晩へ回すには、
    「前回の内省より後」でなく「**まだ畳まれていない**」を材料にしないと、持ち越した分が
    次の晩に消える（`内省` の記録が新しくなるため）。
    """
    now = datetime.now(timezone.utc)
    conn = _conn()
    try:
        with conn.cursor() as cur:
            old = _plant(cur, "前回の内省より前だが畳まれていない", now - timedelta(hours=30))
            _plant(cur, "内省を回した", now - timedelta(hours=24), direction="内省")
            a = _plant(cur, "朝の会話", now - timedelta(hours=20))
            b = _plant(cur, "昼に見たもの", now - timedelta(hours=12), direction="観察")
            core = _plant(cur, "大事な話", now - timedelta(hours=10), n=1)
            folded = _plant(cur, "もう畳んだ", now - timedelta(hours=8), hidden=True)
            req = _plant(cur, "「天気は？」と聞かれ…", now - timedelta(hours=6), direction="求め")
            future = _plant(cur, "この内省が始まった後", now + timedelta(minutes=5))
    finally:
        conn.close()

    rows = _oif().fold_materials(FOLD_DIRECTIONS, before=now)
    ids = [r.obs_id for r in rows]
    assert ids == [old, a, b], ids  # 古い順・内省より前の持ち越しも入る
    for excluded in (core, folded, req, future):
        assert excluded not in ids


def test_without_a_previous_rest_everything_is_material():
    now = datetime.now(timezone.utc)
    conn = _conn()
    try:
        with conn.cursor() as cur:
            a = _plant(cur, "初日の会話", now - timedelta(days=3))
            b = _plant(cur, "二日目", now - timedelta(days=2), direction="独白")
    finally:
        conn.close()
    assert [r.obs_id for r in _oif().fold_materials(FOLD_DIRECTIONS, before=now)] == [a, b]


def test_the_rows_carry_what_the_fold_needs():
    now = datetime.now(timezone.utc)
    conn = _conn()
    try:
        with conn.cursor() as cur:
            _plant(cur, "ももことサッカーの話", now - timedelta(hours=1))
    finally:
        conn.close()
    r = _oif().fold_materials(FOLD_DIRECTIONS, before=datetime.now(timezone.utc))[0]
    assert r.content == "ももことサッカーの話" and r.direction == "発話"
    assert r.timestamp is not None


# ── 書く・畳む（実 DB） ───────────────────────────────────────────────────────


def _agent_for_fold(backend_reply: str):
    from unittest.mock import AsyncMock, MagicMock

    from familiar_agent.io.oif import OIF
    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    a = MagicMock()
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[1.0] + [0.0] * 1023]),
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0] + [0.0] * 1023]),
    ):
        a._memory = ObservationMemory()
    a._oif = OIF(a._memory, for_person=lambda pid: a._memory.for_person(pid))
    a.backend = AsyncMock()
    a.backend.complete = AsyncMock(return_value=backend_reply)
    a._family_md = "## パパ\n名前: いちろう\n\n## ももこ\n名前: ももこ\n"
    a._pmm.find_person_id_by_name = MagicMock(side_effect=lambda n: {"ももこ": "pid-momoko"}.get(n))
    a._pmm.get_present_ids = MagicMock(return_value=[])
    a._observation_perspective = MagicMock(
        return_value={"writer_id": AGENT_SELF_ID, "participants": []}
    )
    a._evaluator.emotion_for_turn = AsyncMock(return_value=(None, 0.3, "neutral"))
    return a


def test_fold_day_writes_the_episode_and_person_summaries_and_folds_the_materials():
    import asyncio

    from familiar_agent.loop import rest_fold

    now = datetime.now(timezone.utc)
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO persons (id, name, display_name, created_at, updated_at) "
                "VALUES ('pid-momoko', 'ももこ', 'ももこ', now()::text, now()::text) ON CONFLICT DO NOTHING"
            )
            a1 = _plant(cur, "ももこ：サッカーしたよ", now - timedelta(hours=5))
            a2 = _plant(
                cur, "自分が答えた：いいね、楽しかった？", now - timedelta(hours=5, minutes=-1)
            )
    finally:
        conn.close()
    agent = _agent_for_fold(
        '{"episode": "今日はももことサッカーの話をした。楽しそうで、ぼくもうれしかった。", '
        '"persons": {"ももこ": "サッカーが好き。今日は楽しそうだった。"}}'
    )
    with patch("familiar_agent.loop.rest_fold.family_names_of", return_value=("パパ", "ももこ")):
        result = asyncio.run(rest_fold.fold_since_last_rest(agent))
    assert result.folded == 2 and result.written == 2 and result.skipped == 0

    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content, direction FROM observations WHERE direction IN ('記憶','人物') ORDER BY timestamp"
            )
            written = cur.fetchall()
            # 関係の表はテスト間で消えないので、自分の材料が「畳み込み」の旧で、新が自己エピソードであることだけを見る。
            cur.execute(
                "SELECT old.obs_id AS old_id, new.obs_id AS new_id FROM relation_members old "
                "JOIN relations r ON r.id = old.relation_id AND r.kind = '畳み込み' "
                "JOIN relation_members new ON new.relation_id = r.id AND new.role = '新' "
                "WHERE old.role = '旧' AND old.obs_id = ANY(%s)",
                ([a1, a2],),
            )
            pairs = {(r["old_id"], r["new_id"]) for r in cur.fetchall()}
            cur.execute(
                "SELECT person_id FROM situated_memories WHERE obs_id = (SELECT id FROM observations WHERE direction='人物' LIMIT 1)"
            )
            facets = {r["person_id"] for r in cur.fetchall()}
    finally:
        conn.close()
    directions = {w["direction"] for w in written}
    assert directions == {"記憶", "人物"}, directions
    episode_id = next(w["id"] for w in written if w["direction"] == "記憶")
    assert pairs == {(a1, episode_id), (a2, episode_id)}
    assert "pid-momoko" in facets  # 関係のまとめはその人の面に立つ


def test_a_rejected_summary_writes_nothing_and_folds_nothing():
    import asyncio

    from familiar_agent.loop import rest_fold

    now = datetime.now(timezone.utc)
    conn = _conn()
    try:
        with conn.cursor() as cur:
            _plant(cur, "何か", now - timedelta(hours=1))
    finally:
        conn.close()
    agent = _agent_for_fold('{"episode": "", "persons": {}}')
    with patch("familiar_agent.loop.rest_fold.family_names_of", return_value=("パパ",)):
        result = asyncio.run(rest_fold.fold_since_last_rest(agent))
    assert result.written == 0 and result.folded == 0 and result.skipped == 1
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM observations WHERE direction='記憶'")
            assert cur.fetchone()["n"] == 0
    finally:
        conn.close()


def test_nothing_to_fold_is_reported():
    import asyncio

    from familiar_agent.loop import rest_fold

    agent = _agent_for_fold("{}")
    result = asyncio.run(rest_fold.fold_since_last_rest(agent))
    assert (result.materials, result.written, result.folded) == (0, 0, 0)


def test_only_the_first_batches_are_asked_in_one_night_and_the_rest_are_deferred():
    """1 晩の回数の上限（記-l）。超えた分は畳まれずに残り、次の晩に古い順で続く。"""
    import asyncio
    from unittest.mock import MagicMock, patch

    from familiar_agent.io.oif import MI
    from familiar_agent.loop import rest_fold

    rows = [
        MI(
            id=f"o{i}",
            obs_id=f"o{i}",
            content=f"記録 {i}",
            timestamp=datetime(2026, 6, 1 + i, 10, tzinfo=timezone.utc),
            direction="発話",
        )
        for i in range(5)
    ]  # 5 日 → 5 回
    agent = MagicMock()
    # 向きごとに返す（聞こえた話は別に、about の期間を過ぎた分だけを引く・記-ae）。
    agent._oif.fold_materials = MagicMock(
        side_effect=lambda directions, *, before: [] if directions == ("聞こえた",) else rows
    )
    agent.backend.complete = MagicMock()

    async def _ask(backend, batch, *, family_names):
        return None  # 見送り（数だけ見る）

    with patch("familiar_agent.loop.rest_fold.ask_summaries", new=_ask):
        result = asyncio.run(rest_fold.fold_since_last_rest(agent, max_batches=2))
    assert result.batches == 2 and result.deferred == 3 and result.skipped == 2
    assert rest_fold.DEFAULT_MAX_BATCHES == 10


# --- 聞こえた話も畳む（記-ae・2026-11-24・本人の決定ア） ------------------------------------


def test_heard_family_talk_is_folded_like_the_other_records():
    """窓の外で聞こえた家族の話（向き `聞こえた`）も、ほかの記録と同じく夜に畳む（溜め続けない）。"""
    from familiar_agent.core.heard import HEARD_DIRECTION
    from familiar_agent.loop.rest_fold import FOLD_DIRECTIONS as LIVE

    assert HEARD_DIRECTION in LIVE


def test_the_rest_pass_reads_who_was_talked_about_before_it_folds():
    """畳むと記録が隠れるので、話題にされた人（about）を読むのは、核の固めと畳み込みより先。

    **反証側**：about が後なら、その晩に聞こえた話は読まれる前に隠れる（`core/heard` がそれまで畳まなかった理由）。
    """
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    from familiar_agent.loop.rest import run_rest_pass
    from familiar_agent.loop.rest_about import AboutResult
    from familiar_agent.loop.rest_core import CoreResult
    from familiar_agent.loop.rest_fold import FoldResult

    order: list[str] = []
    agent = MagicMock()
    agent._oif.write = AsyncMock(return_value="obs1")
    agent._observation_perspective = MagicMock(return_value={})

    async def about(_agent):
        order.append("about")
        return AboutResult(sources=0, asked=0, added=0, unanswered=0)

    async def core(_agent):
        order.append("核")
        return CoreResult()

    async def fold(_agent):
        order.append("畳み込み")
        return FoldResult(materials=0, batches=0, written=0, folded=0, skipped=0)

    with (
        patch("familiar_agent.loop.rest.add_about_tonight", new=about),
        patch("familiar_agent.loop.rest.fold_core", new=core),
        patch("familiar_agent.loop.rest.fold_since_last_rest", new=fold),
    ):
        asyncio.run(run_rest_pass(agent))
    assert order == ["about", "核", "畳み込み"]


def test_heard_talk_is_folded_only_after_the_about_step_stops_looking_at_it():
    """about は直近 `about_lookback_days`（2 日）の聞こえた話を毎晩見直す（Jev が落ちた晩や 1 晩の上限で漏れても、次の晩に読む）。
    畳むと隠れるので、聞こえた話はその期間を過ぎてから畳む。ほかの向きはいままでどおり、いままでの分を畳む。"""
    import asyncio
    from datetime import datetime, timedelta, timezone
    from unittest.mock import MagicMock

    from familiar_agent.config import MemoryConfig
    from familiar_agent.core.heard import HEARD_DIRECTION
    from familiar_agent.loop import rest_fold

    calls: list = []
    agent = MagicMock()
    agent.config.memory.distill_min_a0 = 0.0

    def materials(directions, *, before):
        calls.append((tuple(directions), before))
        return []

    agent._oif.fold_materials = materials
    asyncio.run(rest_fold.fold_since_last_rest(agent))
    now = datetime.now(timezone.utc)
    by_dirs = {d: b for d, b in calls}
    others = tuple(d for d in rest_fold.FOLD_DIRECTIONS if d != HEARD_DIRECTION)
    assert set(by_dirs) == {others, (HEARD_DIRECTION,)}
    assert abs((now - by_dirs[others]).total_seconds()) < 60
    wait = timedelta(days=MemoryConfig().about_lookback_days)
    assert abs((now - wait - by_dirs[(HEARD_DIRECTION,)]).total_seconds()) < 60


def test_the_diary_is_told_heard_talk_was_not_said_to_paju():
    """聞こえた話を、パジュに言われたこと・パジュとの約束として日記に書かない（事実の鍵を抜かないのと同じ理由）。"""
    from familiar_agent.loop import rest_fold

    assert "聞こえた" in rest_fold._PROMPT and "ぼくに言ったことではない" in rest_fold._PROMPT
