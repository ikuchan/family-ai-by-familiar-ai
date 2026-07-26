"""呼びかけに使う名前は、別名の一覧ではなく代表の1つ。

`display_name` は `FAMILY.md` の「呼び方」で、"パパ、いくながさん、ゆうすけ" のように
読点区切りの**別名の一覧**である（`find_person_id_by_name` は割って照合している）。
これをそのまま名前として渡すと、在席の文脈が
`(present :speaker "たいきくん、たいき")` になり、呼びかけも運任せになる（実機で観測）。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from familiar_agent.person_memory_manager import PersonMemoryManager


def _pmm(persons):
    m = MagicMock(spec=PersonMemoryManager)
    m.list_persons = MagicMock(return_value=persons)
    m.get_person_name = lambda pid: PersonMemoryManager.get_person_name(m, pid)
    return m


def test_first_alias_is_used_for_addressing():
    m = _pmm([{"id": "p1", "display_name": "パパ、いくながさん、ゆうすけ"}])
    assert m.get_person_name("p1") == "パパ"


def test_comma_separated_aliases_are_split_too():
    m = _pmm([{"id": "p1", "display_name": "たいきくん, たいき"}])
    assert m.get_person_name("p1") == "たいきくん"


def test_single_name_is_returned_as_is():
    m = _pmm([{"id": "p1", "display_name": "Default Person"}])
    assert m.get_person_name("p1") == "Default Person"


def test_unknown_person_falls_back_to_the_id_prefix():
    m = _pmm([])
    assert m.get_person_name("abcdef1234") == "abcdef12"
