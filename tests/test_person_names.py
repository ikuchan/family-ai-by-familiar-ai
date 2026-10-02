"""人物表と名前（知-af・2026-10-02・本人の決定）。

名前には 3 種類ある：**名前**（`FAMILY.md` の「名前」・人物表の鍵）、**呼びかけ名**（呼び方の先頭）、**呼び方の別名**
（呼び方の一つずつ）。人を指す言葉は、すべて `FAMILY.md`（いまの記述）で名前に直してから、人物表を名前の完全一致で
引く。人物表の `display_name` は最初に登録したときのまま止まっていたので、照らすのには使わない。

段 1：名前の読み取りが改行をまたいで、テンプレートの空の「名前：」から次の行（`- **呼び方**：…`）を名前として
拾い、ゴミの人物ができていた。話者の帳面には、呼び方の一覧そのものが登録されていた。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from familiar_agent.core.parsing import parse_family_md

TEMPLATE_LIKE = """## （名前を変えてください）

- **名前**：
- **呼び方**：パパ、ゆうすけ
- **英字**：
- **関係**：

## たいき

- **名前**：泰輝
- **呼び方**：たいき、たいきくん
"""


def test_an_empty_name_does_not_take_the_next_line():
    got = parse_family_md(TEMPLATE_LIKE)
    assert [m["name"] for m in got] == ["泰輝"]


def test_empty_fields_stay_empty():
    got = parse_family_md(
        "## a\n- **名前**：雄輔\n- **英字**：\n- **関係**：\n- **呼び方**：パパ\n"
    )
    assert got[0]["latin"] == "" and got[0]["relation"] == ""
    assert got[0]["display_name"] == "パパ"


def test_the_speaker_book_gets_the_call_name():
    from familiar_agent.agent import EmbodiedAgent

    a = MagicMock(spec=EmbodiedAgent)
    a._family_md = "## パパ\n- **名前**：雄輔\n- **呼び方**：パパ、ゆうすけ\n"
    a._pmm = MagicMock()
    a._pmm.list_persons = MagicMock(return_value=[])
    a._persons = MagicMock()
    EmbodiedAgent._register_family_from_md(a)
    a._persons.register.assert_called_with("パパ")


# ── 段 2：`FAMILY.md` で名前に直してから、人物表を名前の完全一致で引く ─────────────

FAMILY = """## パパ

- **名前**：雄輔
- **呼び方**：パパ、ゆうすけ

## たいき

- **名前**：泰輝
- **呼び方**：たいき、たいきくん
"""

#: 統合の前の本番と同じ形：古い行（記憶つき）と新しい行が並び、古い行の呼び方は最初に登録したときのまま。
ROWS = [
    {"id": "old-papa", "name": "いくながゆうすけ", "display_name": "パパ、いくながさん、ゆうすけ"},
    {"id": "papa", "name": "雄輔", "display_name": "パパ、ゆうすけ"},
    {"id": "taiki", "name": "泰輝", "display_name": "たいき"},
]


def _pmm(family: str = FAMILY, rows=ROWS):
    from familiar_agent.person_memory_manager import PersonMemoryManager

    base = MagicMock()
    base.list_persons = MagicMock(return_value=list(rows))
    m = PersonMemoryManager(base)
    m.set_family_md(family)
    return m


def test_any_word_for_a_person_finds_their_row_by_name():
    m = _pmm()
    for word in ("雄輔", "パパ", "ゆうすけ"):
        assert m.find_person_id_by_name(word) == "papa", word
    assert m.find_person_id_by_name("たいきくん") == "taiki"


def test_the_order_of_rows_no_longer_decides():
    """以前は古い行（作られた順の先頭）が別名で先に当たり、記憶の行き先が並び順で決まった。"""
    assert _pmm().find_person_id_by_name("パパ") != "old-papa"


def test_an_alias_only_in_the_old_display_name_does_not_count():
    assert _pmm().find_person_id_by_name("いくながさん") is None


def test_a_word_for_two_people_finds_no_one_and_says_so(caplog):
    family = FAMILY + "\n## こうき\n\n- **名前**：光希\n- **呼び方**：こうき、たいき\n"
    with caplog.at_level("ERROR"):
        assert _pmm(family).find_person_id_by_name("たいき") is None
    assert any("2 人以上" in r.message for r in caplog.records)


def test_a_word_not_in_family_finds_no_one():
    assert _pmm().find_person_id_by_name("おばあちゃん") is None


def test_without_family_only_the_exact_name_counts():
    m = _pmm(family="")
    assert m.find_person_id_by_name("雄輔") == "papa"
    assert m.find_person_id_by_name("パパ") is None


# ── 段 3：起動時と `/reload` の検め ─────────────────────────────────────────────


def _agent_with(rows, family):
    from familiar_agent.agent import EmbodiedAgent

    a = MagicMock(spec=EmbodiedAgent)
    a._family_md = family
    a._pmm = MagicMock()
    a._pmm.list_persons = MagicMock(return_value=list(rows))
    a._pmm.register_person = MagicMock(return_value="new-id")
    a._persons = MagicMock()
    EmbodiedAgent._register_family_from_md(a)
    return a


def test_a_renamed_person_makes_no_new_row_and_says_so(caplog):
    """名前を書き換えて、呼び方が既存の行と重なる——新しい行は作らず、統合の手順へ促す。"""
    old = [
        {
            "id": "old-papa",
            "name": "いくながゆうすけ",
            "display_name": "パパ、いくながさん、ゆうすけ",
        }
    ]
    family = "## パパ\n- **名前**：雄輔\n- **呼び方**：パパ、ゆうすけ\n"
    with caplog.at_level("ERROR"):
        a = _agent_with(old, family)
    a._pmm.register_person.assert_not_called()
    assert any("統合" in r.message for r in caplog.records)


def test_a_new_family_member_is_still_registered():
    family = "## こうき\n- **名前**：光希\n- **呼び方**：こうき\n"
    a = _agent_with([{"id": "papa", "name": "雄輔", "display_name": "パパ"}], family)
    a._pmm.register_person.assert_called_once_with("光希", display_name="こうき")


def test_the_call_names_follow_family_md():
    family = "## パパ\n- **名前**：雄輔\n- **呼び方**：おとうさん、パパ\n"
    a = _agent_with([{"id": "papa", "name": "雄輔", "display_name": "パパ、ゆうすけ"}], family)
    a._pmm.update_display_name.assert_called_once_with("papa", "おとうさん、パパ")
    a._pmm.register_person.assert_not_called()


def test_the_same_call_names_do_not_touch_the_row():
    family = "## パパ\n- **名前**：雄輔\n- **呼び方**：パパ\n"
    a = _agent_with([{"id": "papa", "name": "雄輔", "display_name": "パパ"}], family)
    a._pmm.update_display_name.assert_not_called()


def test_a_word_shared_by_two_family_members_is_reported(caplog):
    family = (
        "## たいき\n- **名前**：泰輝\n- **呼び方**：たいき、にいに\n\n"
        "## こうき\n- **名前**：光希\n- **呼び方**：こうき、にいに\n"
    )
    rows = [
        {"id": "t", "name": "泰輝", "display_name": "たいき、にいに"},
        {"id": "k", "name": "光希", "display_name": "こうき、にいに"},
    ]
    with caplog.at_level("ERROR"):
        _agent_with(rows, family)
    assert any("にいに" in r.message and "2 人" in r.message for r in caplog.records)


def test_the_store_rewrites_the_call_names():
    import uuid
    from datetime import datetime, timezone

    from familiar_agent.db import get_db
    from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID
    from familiar_agent.store.context import StoreContext
    from familiar_agent.store.persons import PersonRegistry as PersonStore

    db = get_db()
    store = PersonStore(
        StoreContext(db=db, lock=db.lock, person_id=DEFAULT_PERSON_ID, embedder=None)
    )
    name = f"呼び方の試験 {uuid.uuid4().hex[:8]}"
    pid = store.register_person(name, "パパ")
    assert store.update_display_name(pid, "おとうさん、パパ")
    row = next(p for p in store.list_persons() if p["id"] == pid)
    assert row["display_name"] == "おとうさん、パパ"
    assert datetime.now(timezone.utc)  # 時刻は updated_at が持つ（値は見ない）
