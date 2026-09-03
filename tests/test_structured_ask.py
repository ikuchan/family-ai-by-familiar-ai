"""軽量LLM へ「形のある答え」を求める口（出-c の後段）。

**モデルを替えたとき壊れるのは、ここだった。** `complete()` は文字列を返すだけで、
呼び出し側がそれぞれ解釈していた——数値を正規表現で拾う、集合に含まれるか見る、
`yes` で始まるか見る、JSON として読む。**同じ問題を7箇所が別々に解いており**、
外れたときの扱いも割れていた（記録して倒すもの／黙って既定にするもの）。

**返すのは「取れたか、取れなかったか」だけである。** 既定へ落とすのは呼び出し側の判断で、
口はそれを持たない（050 で「測れなかった PAD を気分で埋めない」と決めたのと同じ形）。

**取れなかったことを数える。** どのモデルがどの形をどれだけ外すかが測れれば、それが
モデル選定の判断材料になる。いまは `.startswith("yes")` が「はい、同じです」を黙って
偽にしていた。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from familiar_agent.core.structured_ask import (
    ask_choice,
    ask_json,
    ask_numbers,
    ask_subset,
    ask_yes_no,
)


def _backend(reply: str):
    b = AsyncMock()
    b.complete = AsyncMock(return_value=reply)
    return b


def _run(coro):
    return asyncio.run(coro)


# ── 数値の並び ──────────────────────────────────────────────────────────────

def test_numbers_are_read_in_order() -> None:
    got = _run(ask_numbers(_backend("0.7 0.2 0.6"), "p", count=3))
    assert got == (0.7, 0.2, 0.6)


def test_numbers_are_clamped_to_the_range() -> None:
    got = _run(ask_numbers(_backend("1.5 -0.1 0.6"), "p", count=3))
    assert got == (1.0, 0.0, 0.6)


def test_too_few_numbers_is_not_taken() -> None:
    """3つ求めて2つしか返らないなら、**取れなかった**。埋めない。"""
    assert _run(ask_numbers(_backend("0.7 0.2"), "p", count=3)) is None


def test_prose_without_numbers_is_not_taken() -> None:
    assert _run(ask_numbers(_backend("I think it's happy!"), "p", count=3)) is None


# ── 選択肢から1つ ───────────────────────────────────────────────────────────

_MOODS = {"engaged", "tired", "frustrated", "absent", "happy"}


def test_a_choice_is_matched_case_insensitively() -> None:
    assert _run(ask_choice(_backend("  Tired\n"), "p", choices=_MOODS)) == "tired"


def test_a_choice_is_found_inside_a_sentence() -> None:
    """モデルは短く答えろと言っても文で返すことがある。含まれていれば拾う。"""
    assert _run(ask_choice(_backend("The person seems tired."), "p", choices=_MOODS)) == "tired"


def test_an_unknown_answer_is_not_taken() -> None:
    """**黙って既定にしない。** 取れなかったことを呼び出し側へ返す。"""
    assert _run(ask_choice(_backend("たぶん元気そう"), "p", choices=_MOODS)) is None


def test_an_ambiguous_answer_is_not_taken() -> None:
    """2つ以上あてはまるなら選べていない。"""
    assert _run(ask_choice(_backend("tired or happy"), "p", choices=_MOODS)) is None


# ── はい／いいえ ────────────────────────────────────────────────────────────

def test_yes_in_english() -> None:
    assert _run(ask_yes_no(_backend("Yes, they are the same."), "p")) is True


def test_yes_in_japanese() -> None:
    """`.startswith("yes")` は「はい、同じです」を**黙って偽**にしていた。"""
    assert _run(ask_yes_no(_backend("はい、同じです。"), "p")) is True


def test_no_in_either_language() -> None:
    assert _run(ask_yes_no(_backend("No."), "p")) is False
    assert _run(ask_yes_no(_backend("いいえ、違います"), "p")) is False


def test_neither_yes_nor_no_is_not_taken() -> None:
    assert _run(ask_yes_no(_backend("わかりません"), "p")) is None


# ── 選択肢の部分集合 ────────────────────────────────────────────────────────

_AXES = {"seeking", "rest", "bond", "safety", "esteem"}


def test_a_subset_picks_every_named_choice() -> None:
    got = _run(ask_subset(_backend("seeking, bond"), "p", choices=_AXES))
    assert got == frozenset({"seeking", "bond"})


def test_an_explicit_none_is_an_empty_subset() -> None:
    """「無ければ none」と聞いている。**空集合は取れた答えであって、失敗ではない。**"""
    assert _run(ask_subset(_backend("none"), "p", choices=_AXES)) == frozenset()


def test_an_unreadable_subset_is_not_taken() -> None:
    assert _run(ask_subset(_backend("よくわかりません"), "p", choices=_AXES)) is None


# ── JSON ───────────────────────────────────────────────────────────────────

def test_json_is_read() -> None:
    assert _run(ask_json(_backend('{"entities": []}'), "p")) == {"entities": []}


def test_json_wrapped_in_a_code_fence_is_read() -> None:
    """実機の VLM は**同じ画像でも回ごとに**包んだり包まなかったりする。"""
    got = _run(ask_json(_backend('```json\n{"entities": [{"label": "cat"}]}\n```'), "p"))
    assert got == {"entities": [{"label": "cat"}]}


def test_broken_json_is_not_taken() -> None:
    assert _run(ask_json(_backend("{entities: "), "p")) is None


def test_a_json_array_is_not_taken_when_an_object_is_expected() -> None:
    assert _run(ask_json(_backend("[1, 2, 3]"), "p")) is None


# ── 共通：呼び出しが失敗しても落とさない ───────────────────────────────────

def test_a_failing_backend_yields_nothing() -> None:
    """軽量LLM が落ちても、口は例外を投げない（呼び出し側が既定を決める）。"""
    b = AsyncMock()
    b.complete = AsyncMock(side_effect=RuntimeError("落ちた"))
    assert _run(ask_yes_no(b, "p")) is None
    assert _run(ask_choice(b, "p", choices=_MOODS)) is None
    assert _run(ask_numbers(b, "p", count=3)) is None
    assert _run(ask_json(b, "p")) is None
    assert _run(ask_subset(b, "p", choices=_AXES)) is None
