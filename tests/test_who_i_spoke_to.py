"""自分の発話が、誰へ言ったものかを引く（出-ak・2026-09-23）。

W の 1 行はいま `わたし：自分が答えた：…` としか書かず、**誰へ言ったかが落ちている**。
そのため子どもへの常体が大人との会話の W に並び、口調を引っぱっていた（実機 17:25）。

誰が居たかは `present` の面（`[そばに居た]`）が持っている。ここはそれを名前へ直す口。
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import psycopg2

from familiar_agent.person_memory_manager import AGENT_SELF_ID

_DIM = 1024


def _vec_sql() -> str:
    v = np.zeros(_DIM, dtype=np.float32)
    v[0] = 1.0
    return "[" + ",".join(f"{x:.1f}" for x in v) + "]"


def _conn():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def _store():
    from unittest.mock import patch

    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory()


def _person(name: str, display: str) -> str:
    pid = str(uuid.uuid4())
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO persons (id, name, display_name, created_at, updated_at) "
                "VALUES (%s,%s,%s,%s,%s)",
                (pid, name, display, "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
            )
        conn.commit()
    finally:
        conn.close()
    return pid


def _plant(body: str, *, present: "list[str]") -> str:
    """自分の発話を 1 件。`present` の面を人数ぶん立てる。"""
    obs_id = str(uuid.uuid4())
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction, kind, emotion) "
                "VALUES (%s,%s,now(),%s,%s,%s)",
                (obs_id, body, "発話", "conversation", "neutral"),
            )
            cur.execute(
                "INSERT INTO situated_memories "
                "(id, obs_id, person_id, vector, relation_key, content) "
                "VALUES (%s,%s,%s,%s::vector,%s,%s)",
                (str(uuid.uuid4()), obs_id, AGENT_SELF_ID, _vec_sql(), "actor", body),
            )
            for pid in present:
                cur.execute(
                    "INSERT INTO situated_memories "
                    "(id, obs_id, person_id, vector, relation_key, content) "
                    "VALUES (%s,%s,%s,%s::vector,%s,%s)",
                    (str(uuid.uuid4()), obs_id, pid, _vec_sql(), "present", f"[そばに居た] {body}"),
                )
        conn.commit()
    finally:
        conn.close()
    return obs_id


def test_the_person_who_was_there_is_the_addressee():
    pid = _person("ゆうすけ", "パパ、ゆうすけ、おとうさん")
    oid = _plant("おかえりなさい", present=[pid])
    got = {k: v[1] for k, v in _store().voices_of([oid]).items()}
    assert got == {oid: ["パパ"]}, "呼びかけに使う名前（呼び方の先頭）で返す"


def test_several_people_come_back_in_one_list():
    a = _person("たいき", "たいき、たいきくん")
    b = _person("こうき", "こうき、こうきくん")
    oid = _plant("おかえり！", present=[a, b])
    got = {k: v[1] for k, v in _store().voices_of([oid]).items()}
    assert sorted(got[oid]) == ["こうき", "たいき"]


def test_myself_is_not_an_addressee():
    """自分の面（`actor`）は相手ではない。"""
    oid = _plant("ひとりごと", present=[])
    assert _store().voices_of([oid]) == {oid: ("わたし", [])}


def test_a_record_without_any_face_is_absent():
    """面が立っていない記録は**入らない**（主体も相手も言わない）。"""
    assert _store().voices_of([str(uuid.uuid4())]) == {}


def test_nothing_in_nothing_out():
    assert _store().voices_of([]) == {}
