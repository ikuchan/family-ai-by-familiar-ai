"""畳み込みは家族の呼び方を一つずつに分けて使う（2026-10-08 実機 04:25）。

`family_names_of` は `FAMILY.md` の「呼び方」（「パパ、いちろう、いっちゃんさん、いちくん、いっくん」）を分けずに 1 つの名前として
返していた。返りの「パパ」「さぶろう」を「家族に無い名前」として弾き、一晩 10 回の枠のうち 6 回を見送った（畳めたのは
4,400 件中 133 件）。その日に出てきた家族を拾う `_participants` も、長い文字列を本文から探していたので誰も拾えなかった。
分け方は `core/speaker_claim`（`aliases_of`・`call_name_of`）にそろえる。ここでは差し替えずに、本物の読み込みを通す。
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from familiar_agent.loop import rest_fold

FAMILY_MD = """# 一緒に暮らす人たち

## いちろう

- **名前**：一郎
- **呼び方**：パパ、いちろう、いっちゃんさん、いちくん、いっくん

## じろう

- **名前**：次郎
- **呼び方**：ママ、じろう、じっちゃんさん、じろちゃん、じーくん

## さぶろう

- **名前**：三郎
- **呼び方**：さぶろう、さぶちゃん、さぶくん、Saburo

## ももこ

- **名前**：桃子
- **呼び方**：ももこ,ももちゃん,Momoko
"""

_IDS = {"パパ": "pid-papa", "一郎": "pid-papa", "ももこ": "pid-momoko", "さぶろう": "pid-saburo"}


def _agent():
    pmm = SimpleNamespace(find_person_id_by_name=lambda name: _IDS.get(name))
    return SimpleNamespace(_family_md=FAMILY_MD, _pmm=pmm)


def test_every_way_of_calling_is_a_family_name():
    names = rest_fold.family_names_of(_agent())
    for word in ("パパ", "いちろう", "一郎", "ママ", "さぶろう", "ももこ", "Momoko"):
        assert word in names
    assert not any("、" in n or "," in n for n in names)


def test_the_check_lets_the_names_of_that_night_through():
    """04:25 に弾かれた 3 つの名前（パパ・ママ・さぶろう）が通る。"""
    names = rest_fold.family_names_of(_agent())
    s = rest_fold.Summaries(
        episode="今日はパパとママとさぶろうと話した。",
        persons={"パパ": "早起き。", "ママ": "料理上手。", "さぶろう": "ゲームが好き。"},
    )
    assert rest_fold.check(s, family_names=names) is None
    stranger = rest_fold.Summaries(episode="今日は話した。", persons={"たなか": "隣の人。"})
    assert rest_fold.check(stranger, family_names=names) is not None


def test_the_request_lists_one_call_name_per_person():
    assert rest_fold.family_call_names_of(_agent()) == ("パパ", "ママ", "さぶろう", "ももこ")
    src = inspect.getsource(rest_fold.fold_since_last_rest)
    assert "family_names=family_call_names_of(agent)" in src


def test_the_family_of_the_day_is_picked_from_the_records():
    batch = rest_fold.Batch(
        day="2026-06-15",
        rows=[
            SimpleNamespace(content="パパ：ただいま"),
            SimpleNamespace(content="ももこが来た"),
        ],
    )
    assert rest_fold._participants(_agent(), batch) == ["pid-papa", "pid-momoko"]


# ── 段 2：呼び方の分け方は `core/speaker_claim` の 1 か所だけ ───────────────────────


def test_the_call_name_comes_from_one_place():
    from familiar_agent import agent
    from familiar_agent.core import family_age
    from familiar_agent.loop import rest_family_now
    from familiar_agent.store import person_merge

    for fn in (
        rest_family_now._members,
        family_age.ages_lines,
        agent.EmbodiedAgent._register_family_from_md,
        person_merge._aliases,
    ):
        src = inspect.getsource(fn)
        assert '.split("、")' not in src, fn.__qualname__
        assert "call_name_of" in src or "aliases_of" in src, fn.__qualname__


def test_a_comma_list_gives_its_first_word():
    """「,」で区切った呼び方でも、先頭の一語を呼び方にする（以前は「、」だけで分けていた）。"""
    from datetime import date

    from familiar_agent.core import family_age

    md = "## ももこ\n\n- **名前**：桃子\n- **呼び方**：ももこ,Momoko\n- **誕生日**：2015-04-10\n"
    (line,) = family_age.ages_lines(md, date(2026, 10, 8))
    assert line.startswith("ももこ：")
