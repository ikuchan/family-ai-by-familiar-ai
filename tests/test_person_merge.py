"""人物表の統合（知-af 段 4・2026-10-02・本人の決定ア）。本番で使う道具 `scripts/merge_persons.py` の中身。

本番の人物表は二重になっている：記憶の付いた古い行（いくながゆうすけ など）と、いまの `FAMILY.md` が作った記憶 0 件の
新しい行（ゆうすけ など）。**記憶の付いた古い行を残し**、名前を `FAMILY.md` の名前（漢字）に書き換え、記憶 0 件の行と
テンプレートから作られたゴミの行を消す。記憶の付け替えはしない。

- 参照は決め打ちにしない。人物表を参照する外部キーを DB から読み、人物ごとに数える。参照が 1 件でもある行は消さない。
- 1 人の家族に参照のある行が 2 つ以上あれば止まる（統合ではなく付け替えが要る）。1 つの行が 2 人の家族に当たっても止まる。
- 計画は純関数で立て、実行は 1 つのトランザクションで、消す → 名前を書き換える の順（名前は一意なので）。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from familiar_agent.core.parsing import parse_family_md
from familiar_agent.store import person_merge as pm

FAMILY = """## パパ

- **名前**：雄輔
- **呼び方**：パパ、ゆうすけ

## たいき

- **名前**：泰輝
- **呼び方**：たいき、たいきくん
"""

ROWS = [
    {"id": "old-papa", "name": "いくながゆうすけ", "display_name": "パパ、いくながさん、ゆうすけ"},
    {"id": "old-taiki", "name": "いくながたいき", "display_name": "たいき"},
    {"id": "new-papa", "name": "ゆうすけ", "display_name": "パパ、ゆうすけ"},
    {"id": "new-taiki", "name": "たいき", "display_name": "たいき"},
    {
        "id": "junk",
        "name": "- **呼び方**：パパ、ゆうすけ",
        "display_name": "- **呼び方**：パパ、ゆうすけ",
    },
]
REFS = {"old-papa": 3585, "old-taiki": 643}


def _plan(rows=ROWS, refs=REFS, family=FAMILY):
    return pm.make_plan(rows, refs, parse_family_md(family))


def test_the_old_rows_with_memories_are_kept_and_renamed():
    plan = _plan()
    assert not plan.stops
    assert {(r.id, r.new_name) for r in plan.renames} == {
        ("old-papa", "雄輔"),
        ("old-taiki", "泰輝"),
    }
    papa = next(r for r in plan.renames if r.id == "old-papa")
    assert papa.new_display == "パパ、ゆうすけ"  # 呼び方も FAMILY.md に合わせる


def test_the_empty_rows_and_the_junk_are_deleted():
    plan = _plan()
    assert {d.id for d in plan.deletes} == {"new-papa", "new-taiki", "junk"}


def test_a_row_with_references_is_never_deleted_but_reassigned():
    """参照のある行は消さない。同じ人に参照のある行がもう 1 つあれば、付け替えてから消す（2026-10-05 に止まるから改めた）。"""
    refs = {**REFS, "new-papa": 1}
    plan = _plan(refs=refs)
    assert "new-papa" not in {d.id for d in plan.deletes}
    assert [(r.from_id, r.to_id) for r in plan.reassigns] == [("new-papa", "old-papa")]
    assert not plan.stops


def test_two_referenced_rows_for_one_person_are_reassigned_into_the_bigger_one():
    rows = ROWS + [{"id": "older", "name": "ゆうすけ旧", "display_name": "パパ"}]
    plan = _plan(rows=rows, refs={**REFS, "older": 5})
    assert ("older", "old-papa") in {(r.from_id, r.to_id) for r in plan.reassigns}
    assert ("old-papa", "雄輔") in {(r.id, r.new_name) for r in plan.renames}
    assert not plan.stops


def test_a_row_matching_two_people_stops_the_plan():
    rows = ROWS + [{"id": "both", "name": "なぞ", "display_name": "パパ、たいき"}]
    plan = _plan(rows=rows, refs={**REFS, "both": 1})
    assert any("なぞ" in s for s in plan.stops)


def test_a_row_already_right_is_left_alone():
    rows = [{"id": "papa", "name": "雄輔", "display_name": "パパ、ゆうすけ"}]
    plan = _plan(rows=rows, refs={"papa": 10}, family=FAMILY.split("## たいき")[0])
    assert not plan.renames and not plan.deletes and not plan.stops


def test_an_unrelated_row_with_memories_is_left_and_reported():
    rows = ROWS + [{"id": "guest", "name": "おきゃくさん", "display_name": "おきゃくさん"}]
    plan = _plan(rows=rows, refs={**REFS, "guest": 2})
    assert "guest" not in {d.id for d in plan.deletes}
    assert any("おきゃくさん" in n for n in plan.notes)


def test_the_agent_and_default_rows_are_never_touched():
    from familiar_agent.person_memory_manager import AGENT_SELF_ID, DEFAULT_PERSON_ID

    rows = ROWS + [
        {"id": AGENT_SELF_ID, "name": "self", "display_name": "self"},
        {"id": DEFAULT_PERSON_ID, "name": "default", "display_name": "Default Person"},
    ]
    plan = _plan(rows=rows)
    touched = {d.id for d in plan.deletes} | {r.id for r in plan.renames}
    assert AGENT_SELF_ID not in touched and DEFAULT_PERSON_ID not in touched


# ── DB（テスト DB）：参照を外部キーから数え、計画を 1 つのトランザクションで当てる ─────────


@pytest.fixture
def conn():
    import psycopg2
    import psycopg2.extras

    from familiar_agent.db import get_db

    get_db()  # マイグレーションを当てておく
    import os

    c = psycopg2.connect(os.environ["DATABASE_URL"], cursor_factory=psycopg2.extras.RealDictCursor)
    yield c
    c.rollback()
    c.close()


def _person(cur, pid: str, name: str, display: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    cur.execute(
        "INSERT INTO persons (id, name, display_name, created_at, updated_at) VALUES (%s,%s,%s,%s,%s)",
        (pid, name, display, now, now),
    )


def _memory_of(cur, pid: str) -> None:
    oid = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO observations (id, content, timestamp, direction, kind) VALUES (%s,'x',%s,'記憶','observation')",
        (oid, datetime.now(timezone.utc)),
    )
    vec = "[" + ",".join(["0.03125"] * 1024) + "]"
    cur.execute(
        "INSERT INTO situated_memories (id, obs_id, person_id, relation_key, vector, content) "
        "VALUES (%s,%s,%s,'present',%s::vector,'x')",
        (str(uuid.uuid4()), oid, pid, vec),
    )


def test_references_are_counted_from_the_foreign_keys(conn):
    mark = uuid.uuid4().hex[:6]
    with conn.cursor() as cur:
        _person(cur, f"a-{mark}", f"統合の試験A {mark}", "")
        _person(cur, f"b-{mark}", f"統合の試験B {mark}", "")
        _memory_of(cur, f"a-{mark}")
        _memory_of(cur, f"a-{mark}")
    refs = pm.references(conn)
    assert refs.get(f"a-{mark}") == 2
    assert refs.get(f"b-{mark}", 0) == 0
    assert "situated_memories" in pm.referencing_tables(conn)


def test_the_plan_is_applied_in_one_transaction(conn):
    mark = uuid.uuid4().hex[:6]
    family = f"## パパ\n- **名前**：雄輔{mark}\n- **呼び方**：パパ{mark}\n"
    with conn.cursor() as cur:
        _person(cur, f"old-{mark}", f"いくながゆうすけ{mark}", f"パパ{mark}")
        _person(cur, f"new-{mark}", f"雄輔{mark}", f"パパ{mark}")
        _memory_of(cur, f"old-{mark}")
    rows = [r for r in pm.persons(conn) if mark in r["name"]]
    plan = pm.make_plan(rows, pm.references(conn), parse_family_md(family))
    assert not plan.stops
    pm.apply_plan(conn, plan)
    after = {r["id"]: r for r in pm.persons(conn) if mark in r["id"]}
    assert set(after) == {f"old-{mark}"}
    assert after[f"old-{mark}"]["name"] == f"雄輔{mark}"
    assert pm.references(conn).get(f"old-{mark}") == 1  # 記憶はそのまま


# ── 付け替えと緩めた対応づけ（2026-10-05・本番でママの記憶が 2 つの行に分かれた）────────────
#
# 本番の統合で、ママの記憶の付いた行が 2 つあった——`いくながたえこ`（呼び方「まま、たえこさん」・43 件）と `たえこ`
# （「ママ」・12 件）。呼び方の完全一致だけで比べていたので、ひらがなの「まま」とカタカナの「ママ」、「たえこさん」と
# 「たえこ」が当たらず、`いくながたえこ` は「誰にも当たらない。残す」に回った。比べる前にカタカナをひらがなに直し、
# 敬称（さん・くん・ちゃん）を落とす。1 人に記憶の付いた行が 2 つ以上あれば、止まらずに付け替える——残すのは
# FAMILY.md の名前になっている行、無ければ記憶の多い行。

MAMA_ROWS = [
    {"id": "old-mama", "name": "いくながたえこ", "display_name": "まま、たえこさん"},
    {"id": "mama", "name": "妙子", "display_name": "ママ、たえこ"},
]
MAMA = "## ママ\n- **名前**：妙子\n- **呼び方**：ママ、たえこ、おかあさん\n"


def test_kana_and_honorifics_do_not_hide_the_same_person():
    plan = _plan(rows=MAMA_ROWS, refs={"old-mama": 43, "mama": 12}, family=MAMA)
    assert not plan.notes and not plan.stops
    assert [(r.from_id, r.to_id) for r in plan.reassigns] == [("old-mama", "mama")]
    assert plan.renames == [] or all(r.id == "mama" for r in plan.renames)


def test_an_unrelated_row_still_matches_no_one():
    rows = MAMA_ROWS + [{"id": "guest", "name": "おきゃくさん", "display_name": "おきゃくさん"}]
    plan = _plan(rows=rows, refs={"old-mama": 43, "mama": 12, "guest": 2}, family=MAMA)
    assert any("おきゃくさん" in n for n in plan.notes)


def test_reassigning_moves_the_memories_and_drops_the_duplicates(conn):
    mark = uuid.uuid4().hex[:6]
    family = f"## ママ\n- **名前**：妙子{mark}\n- **呼び方**：ママ{mark}\n"
    with conn.cursor() as cur:
        _person(cur, f"old-{mark}", f"いくながたえこ{mark}", f"まま{mark}")
        _person(cur, f"new-{mark}", f"妙子{mark}", f"ママ{mark}")
        _memory_of(cur, f"old-{mark}")
        _memory_of(cur, f"old-{mark}")
        _memory_of(cur, f"new-{mark}")
        # 同じ記録が両方の行に同じ関係で付いている（付け替えるとぶつかる組）
        cur.execute(
            "SELECT obs_id FROM situated_memories WHERE person_id = %s LIMIT 1", (f"new-{mark}",)
        )
        shared = cur.fetchone()["obs_id"]
        vec = "[" + ",".join(["0.03125"] * 1024) + "]"
        cur.execute(
            "INSERT INTO situated_memories (id, obs_id, person_id, relation_key, vector, content) "
            "VALUES (%s,%s,%s,'present',%s::vector,'x')",
            (str(uuid.uuid4()), shared, f"old-{mark}", vec),
        )
    rows = [r for r in pm.persons(conn) if mark in r["name"]]
    refs = pm.references(conn)
    assert refs[f"old-{mark}"] == 3 and refs[f"new-{mark}"] == 1
    plan = pm.make_plan(rows, refs, parse_family_md(family))
    assert [(r.from_id, r.to_id) for r in plan.reassigns] == [(f"old-{mark}", f"new-{mark}")]
    stats = pm.apply_plan(conn, plan)
    assert stats[f"old-{mark}"] == {"moved": 2, "dropped": 1}
    after = {r["id"] for r in pm.persons(conn) if mark in r["id"]}
    assert after == {f"new-{mark}"}
    assert pm.references(conn).get(f"new-{mark}") == 3  # 1 ＋ 移した 2（ぶつかった 1 は捨てた）
