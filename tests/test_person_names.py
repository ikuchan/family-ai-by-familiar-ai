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
