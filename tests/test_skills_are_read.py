"""パジュ用のスキル（やり方）を読む（環-au 段 A・2026-10-25・本人の決定）。

スキルは人が `skills/` に 1 つずつ md で書く。1 行目が見出し（名前）、2 行目が「いつ要る：」（Jev にはこの行だけを見せる）、
3 行目からが本文（要るときに書き手と主LLM に載せる）。判断のたびに読み直すので、直したらすぐ効く。
試験は一時フォルダで読む。本物の `skills/` を読むのは、置いた下書きが読める形かを確かめる 1 件だけ。
"""

from __future__ import annotations

import logging

from familiar_agent.core.skills import Skill, load_skills


def _put(root, name: str, text: str) -> None:
    (root / name).write_text(text, encoding="utf-8")


def test_each_file_is_a_name_a_when_and_a_body(tmp_path):
    _put(
        tmp_path,
        "a.md",
        "# しりとり\nいつ要る：しりとりをしているとき\n\n- 「ん」で終わったら負け。\n",
    )
    _put(
        tmp_path, "b.md", "# マジカルバナナ\nいつ要る：マジカルバナナをしているとき\n- 連想する。\n"
    )
    got = load_skills(tmp_path)
    assert got == [
        Skill("しりとり", "しりとりをしているとき", "- 「ん」で終わったら負け。"),
        Skill("マジカルバナナ", "マジカルバナナをしているとき", "- 連想する。"),
    ]


def test_a_file_without_when_is_skipped_with_a_warning(tmp_path, caplog):
    """**反証の側**：「いつ要る：」が無ければ、Jev に聞けないので読み飛ばす（黙って捨てない）。"""
    _put(tmp_path, "a.md", "# しりとり\n- 「ん」で終わったら負け。\n")
    with caplog.at_level(logging.WARNING):
        assert load_skills(tmp_path) == []
    assert "a.md" in caplog.text


def test_no_folder_means_no_skills(tmp_path):
    assert load_skills(tmp_path / "無い") == []


def test_other_files_are_not_skills(tmp_path):
    _put(tmp_path, "メモ.txt", "# x\nいつ要る：y\nz")
    assert load_skills(tmp_path) == []


def test_the_drafts_are_in_place():
    """置いた下書きが、読める形になっている（本物の `skills/` を形だけ確かめる・中身は見ない）。"""
    from familiar_agent.core.skills import SKILLS_DIR

    names = {s.name for s in load_skills(SKILLS_DIR)}
    assert {"しりとり", "マジカルバナナ"} <= names


# --- 見直しの指摘（1 回目）---


def test_hand_written_variations_are_read(tmp_path):
    """BOM・空行・半角のコロンでも読む（人が手で書く形）。"""
    (tmp_path / "a.md").write_text(
        "# しりとり\n\nいつ要る: 遊ぶとき\n\n- 決まり\n", encoding="utf-8-sig"
    )
    assert load_skills(tmp_path) == [Skill("しりとり", "遊ぶとき", "- 決まり")]


def test_a_broken_file_does_not_stop_the_others(tmp_path, caplog):
    (tmp_path / "a.md").write_bytes("# しりとり\nいつ要る：遊ぶとき\n- 決まり".encode("shift_jis"))
    _put(tmp_path, "b.md", "# マジカルバナナ\nいつ要る：遊ぶとき\n- 決まり\n")
    _put(tmp_path, "c.md", "# 空\nいつ要る：遊ぶとき\n")
    with caplog.at_level(logging.WARNING):
        assert [s.name for s in load_skills(tmp_path)] == ["マジカルバナナ"]
    assert "a.md" in caplog.text and "c.md" in caplog.text


def test_the_same_name_twice_keeps_the_first(tmp_path, caplog):
    _put(tmp_path, "a.md", "# しりとり\nいつ要る：遊ぶとき\n- 一つ目\n")
    _put(tmp_path, "b.md", "# しりとり\nいつ要る：遊ぶとき\n- 二つ目\n")
    with caplog.at_level(logging.WARNING):
        assert [s.body for s in load_skills(tmp_path)] == ["- 一つ目"]
    assert "b.md" in caplog.text
