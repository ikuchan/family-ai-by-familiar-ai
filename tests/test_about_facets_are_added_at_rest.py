"""記-ae 段 2：ある人について話したことに、REST がその人の about の面を足す（段②の about だけ・2026-11-22）。

面は最終版に立て、判定は起点（人の言葉）でする。起点はその晩の畳み込みで隠れ、想起に上がらないため。

候補はコードで出す（本文に呼び名が出る登録の人から、話した本人とパジュを除く）。決めるのは Jev（偽の返り）。
面の言葉は「[自分のこと] ＋本文」、ベクトルは記録の埋め込み。足した面は、その人の想起の母集合に入る。
"""

from __future__ import annotations

import asyncio
import os
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import psycopg2

from familiar_agent.core import jev_judges
from familiar_agent.loop import rest_about
from familiar_agent.person_memory_manager import AGENT_SELF_ID

PAPA = {"id": "p-papa", "name": "たろう", "display_name": "パパ、たろう"}
MAMA = {"id": "p-mama", "name": "はなこ", "display_name": "ママ、はなこ"}


class _Jev:
    available = True

    def __init__(self, p):
        self.p = p
        self.states: list[str] = []

    async def ask(self, state, questions):
        self.states.append(state)
        p = self.p(state) if callable(self.p) else self.p
        if p is None:
            return SimpleNamespace(ok=False, answers={})
        return SimpleNamespace(ok=True, answers={"about": {"noul": p}})


class _Oif:
    def __init__(self, rows, roles=None, writes=True):
        self.rows = rows
        self.role_of = roles or {}
        self.writes = writes
        self.read_cap: list[int] = []
        self.added: list[tuple[str, str]] = []

    def person_names(self, ids):
        return {k: v for k, v in {"p-papa": "パパ", "p-mama": "ママ"}.items() if k in ids}

    def roles(self, obs_ids):
        return {k: v for k, v in self.role_of.items() if k in obs_ids}

    def persons(self):
        return [PAPA, MAMA, {"id": AGENT_SELF_ID, "name": "__self__", "display_name": "Agent self"}]

    def about_pending(self, limit, *, since, until=None):
        self.read_cap.append(limit)
        return self.rows[:limit]

    def add_about(self, obs_id, person_id):
        if not self.writes:
            return False
        self.added.append((obs_id, person_id))
        return True


def _run(oif, jev, limit=200):
    return asyncio.run(rest_about.add_about(oif, jev, limit=limit, min_p=0.7, since=None))


def test_a_person_talked_about_gets_an_about_facet():
    oif = _Oif([{"obs_id": "o1", "content": "パパ、最近忙しそうなんだよね", "actor": "p-mama"}])
    jev = _Jev(0.78)
    got = _run(oif, jev)
    assert oif.added == [("o1", "p-papa")]  # 話した本人（ママ）は候補にしない
    assert (got.sources, got.asked, got.added) == (1, 1, 1)
    assert "[話した人]\nママ" in jev.states[0] and "[この人]\nパパ" in jev.states[0]
    assert "[話しかけた相手]\nロボット（パジュ）" in jev.states[0]


def test_a_call_is_not_an_about():
    oif = _Oif([{"obs_id": "o1", "content": "パパ、ごはんできたよ", "actor": "p-mama"}])
    got = _run(oif, _Jev(0.13))
    assert oif.added == [] and got.asked == 1 and got.added == 0


def test_no_answer_adds_nothing_and_is_counted():
    oif = _Oif([{"obs_id": "o1", "content": "ママ怒ると怖いんだよ", "actor": "p-papa"}])
    got = _run(oif, _Jev(None))
    assert oif.added == [] and got.unanswered == 1


def test_a_yes_that_cannot_be_written_is_counted():
    oif = _Oif([{"obs_id": "o1", "content": "パパは出張", "actor": "p-mama"}], writes=False)
    got = _run(oif, _Jev(0.9))
    assert got.added == 0 and got.unwritten == 1


def test_the_question_cap_does_not_cap_the_rows_read():
    """問いの上限で、読む発話を減らさない（名前の出ない発話や、パジュの返事で上限を使い切らない）。"""
    oif = _Oif([{"obs_id": "o1", "content": "雨だね", "actor": "p-mama"}])
    asyncio.run(rest_about.add_about(oif, _Jev(0.9), limit=3, min_p=0.7, since=None, rows=50))
    assert oif.read_cap == [50]


def test_a_person_already_with_a_facet_is_not_asked_again():
    oif = _Oif(
        [{"obs_id": "o1", "content": "パパとママが出かけた", "actor": None, "done": ["p-papa"]}],
        roles={"o1": "起点"},
    )
    jev = _Jev(0.9)
    _run(oif, jev)
    assert oif.added == [("o1", "p-mama")] and len(jev.states) == 1


def test_an_unknown_speaker_is_told_as_unknown():
    """話者が解決できなかった人の言葉（actor は自分の印・役割は起点）は調べ、話した人を「分からない人」と渡す。"""
    oif = _Oif(
        [{"obs_id": "o1", "content": "パパは明日から出張", "actor": AGENT_SELF_ID}],
        roles={"o1": "起点"},
    )
    jev = _Jev(0.9)
    _run(oif, jev)
    assert "[話した人]\n（分からない人）" in jev.states[0]


def test_without_a_name_in_the_words_nothing_is_asked():
    oif = _Oif([{"obs_id": "o1", "content": "今日は雨だね", "actor": "p-mama"}])
    jev = _Jev(0.9)
    got = _run(oif, jev)
    assert jev.states == [] and got.asked == 0


def test_questions_stop_at_the_nightly_limit():
    rows = [{"obs_id": f"o{i}", "content": "パパとママが出かけた", "actor": None} for i in range(5)]
    oif = _Oif(rows, roles={f"o{i}": "起点" for i in range(5)})
    got = _run(oif, _Jev(0.9), limit=3)
    assert got.asked == 3 and len(oif.added) == 3 and got.sources == 2  # 調べた発話だけ数える


def test_without_jev_it_skips():
    got = _run(_Oif([]), SimpleNamespace(available=False))
    assert got.skipped and got.asked == 0


def test_the_boundary_is_the_given_probability():
    jev = _Jev(0.7)
    assert (
        asyncio.run(jev_judges.judge_about(jev, text="x", speaker="a", person="b", min_p=0.7))
        is True
    )
    jev = _Jev(0.69)
    assert (
        asyncio.run(jev_judges.judge_about(jev, text="x", speaker="a", person="b", min_p=0.7))
        is False
    )


# ── 本物の記憶の箱（試験 DB）─────────────────────────────────────

_DIM = 1024


def _conn():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def _store():
    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory()


def _plant(body: str, *, actor: str) -> "tuple[str, str]":
    """人の言葉の起点（向き 発話・話した人の面）と、それを親に持つ最終版（埋め込みつき）を 1 組。(起点, 最終版) を返す。"""
    from familiar_agent.store.embedding import _encode_vector

    origin, final = str(uuid.uuid4()), str(uuid.uuid4())
    vec = np.zeros(_DIM, dtype=np.float32)
    vec[3] = 1.0
    sql_vec = "[" + ",".join(f"{x:.1f}" for x in vec) + "]"
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO persons (id, name, display_name, created_at, updated_at) "
                "VALUES (%s,%s,%s,now(),now()) ON CONFLICT DO NOTHING",
                (actor, "はなこ", "ママ"),
            )
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction, emotion) "
                "VALUES (%s,%s,now(),'発話','neutral')",
                (origin, body),
            )
            cur.execute(
                "INSERT INTO situated_memories (id, obs_id, person_id, vector, relation_key) "
                "VALUES (%s,%s,%s,%s::vector,'actor')",
                (str(uuid.uuid4()), origin, actor, sql_vec),
            )
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction, emotion, parent_id) "
                "VALUES (%s,%s,now(),'最終版','neutral',%s)",
                (
                    final,
                    f"〔最終版 2026-11-22 10:00:00〜10:00:05〕「{body}」と聞かれ、「そうなんだ」と答えた。",
                    origin,
                ),
            )
            cur.execute(
                "INSERT INTO obs_embeddings (obs_id, vector) VALUES (%s,%s)",
                (final, _encode_vector(vec)),
            )
            cur.execute(
                "INSERT INTO situated_memories (id, obs_id, person_id, vector, relation_key) "
                "VALUES (%s,%s,%s,%s::vector,'actor')",
                (str(uuid.uuid4()), final, AGENT_SELF_ID, sql_vec),
            )
        conn.commit()
    finally:
        conn.close()
    return origin, final


def test_the_store_writes_an_about_facet_and_tells_who_already_has_one():
    mama = str(uuid.uuid4())
    papa = str(uuid.uuid4())
    _origin, oid = _plant("パパ、最近忙しそうなんだよね", actor=mama)
    from datetime import datetime, timedelta, timezone

    since = datetime.now(timezone.utc) - timedelta(days=2)
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO persons (id, name, display_name, created_at, updated_at) "
                "VALUES (%s,%s,%s,now(),now())",
                (papa, "たろう", "パパ"),
            )
            cur.execute(
                "DELETE FROM embedding_means"
            )  # 中心化の平均が無ければ、面のベクトルは記録の埋め込みのまま
        conn.commit()
    finally:
        conn.close()
    store = _store()
    pending = store.about_pending(10, since=since)
    # 最終版を、起点の言葉と、起点を話した人で渡す
    assert [(r["obs_id"], r["content"], r["actor"], r["done"]) for r in pending] == [
        (oid, "パパ、最近忙しそうなんだよね", mama, [])
    ]
    assert store.add_about(oid, papa) is True
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT content, (vector <=> (SELECT vector FROM situated_memories "
                " WHERE obs_id = %s AND relation_key = 'actor')) FROM situated_memories "
                "WHERE obs_id = %s AND person_id = %s AND relation_key = 'about'",
                (oid, oid, papa),
            )
            content, distance = cur.fetchone()
    finally:
        conn.close()
    assert content.startswith("[自分のこと] 〔最終版 ")  # 面の言葉は、見え続ける最終版の本文
    assert distance < 1e-6  # 記録の埋め込みと同じベクトル
    assert [r["done"] for r in store.about_pending(10, since=since)] == [
        [papa]
    ]  # 面の立った人は聞き直さない


def test_overheard_family_talk_is_judged_as_said_to_family():
    """窓の外で聞こえた家族の話（向き 聞こえた・記-ae 段 4）は、話しかけた相手を家族として聞く。"""
    oif = _Oif(
        [
            {
                "obs_id": "h1",
                "content": "パパは明日から出張なんだって",
                "actor": "p-mama",
                "heard": True,
            }
        ]
    )
    jev = _Jev(0.89)
    _run(oif, jev)
    assert oif.added == [("h1", "p-papa")]
    assert "[話しかけた相手]\n家族のだれか（ロボットではない）" in jev.states[0]


def test_the_store_offers_heard_records_with_their_own_words(tmp_path):
    from datetime import datetime, timedelta, timezone

    mama, papa = str(uuid.uuid4()), str(uuid.uuid4())
    heard = str(uuid.uuid4())
    vec = np.zeros(_DIM, dtype=np.float32)
    vec[5] = 1.0
    sql_vec = "[" + ",".join(f"{x:.1f}" for x in vec) + "]"
    conn = _conn()
    try:
        with conn.cursor() as cur:
            for pid, name, call in ((mama, "はなこ", "ママ"), (papa, "たろう", "パパ")):
                cur.execute(
                    "INSERT INTO persons (id, name, display_name, created_at, updated_at) "
                    "VALUES (%s,%s,%s,now(),now())",
                    (pid, name, call),
                )
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction, emotion) "
                "VALUES (%s,%s,now(),'聞こえた','neutral')",
                (heard, "パパは明日から出張なんだって"),
            )
            cur.execute(
                "INSERT INTO situated_memories (id, obs_id, person_id, vector, relation_key) "
                "VALUES (%s,%s,%s,%s::vector,'actor')",
                (str(uuid.uuid4()), heard, mama, sql_vec),
            )
        conn.commit()
    finally:
        conn.close()
    since = datetime.now(timezone.utc) - timedelta(days=2)
    rows = _store().about_pending(10, since=since)
    assert [(r["obs_id"], r["content"], r["actor"], r["heard"]) for r in rows] == [
        (heard, "パパは明日から出張なんだって", mama, True)
    ]
