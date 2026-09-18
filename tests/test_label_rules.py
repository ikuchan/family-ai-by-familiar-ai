"""タイマー・ストップウォッチの名前の検め（知-u の残り・2026-09-19）。

`label` は LLM が書く。09-16 に「何のために時間を測るか不明」が名前として入り、枠・返事・O の記録にそのまま出た。
名前は物の名であるべきなので機械で検める（`core/label_rules.clean_label`・純関数・両方で共通）：
前後の空白・引用符・句点を落とす／20 字〔仮〕を超える・「不明」「わからない」などを含む → 既定名。
"""

from __future__ import annotations

import asyncio

from familiar_agent.core.label_rules import MAX_CHARS, clean_label


def test_short_names_pass_after_trimming():
    assert MAX_CHARS == 20
    assert clean_label("お風呂", "測る") == "お風呂"
    assert clean_label("「パパの頼まれごと」。 ", "測る") == "パパの頼まれごと"
    assert (
        clean_label("パパがコーヒーを淹れる時間", "測る") == "パパがコーヒーを淹れる時間"
    )  # 13 字


def test_long_or_unknown_names_fall_back():
    assert clean_label("何のために時間を測るか不明", "測る") == "測る"
    assert clean_label("わからないけど測る", "測る") == "測る"
    assert clean_label("特になし", "タイマー") == "タイマー"
    assert (
        clean_label("パパがキッチンでパスタを茹でるのにかかる時間を測る", "測る") == "測る"
    )  # 20 字超
    assert clean_label("", "タイマー") == "タイマー"


def test_the_tools_apply_it():
    from tests.test_stopwatch import _tool as _sw
    from tests.test_timer_tool import _tool as _timer

    t, store, _ = _sw()
    asyncio.run(t.call("start_stopwatch", {"label": "何のために時間を測るか不明"}))
    assert store.rows[0]["label"] == "測る"
    t2, store2, _ = _timer()
    asyncio.run(t2.call("set_timer", {"after_minutes": 3, "label": "  「不明」 "}, confirmed=True))
    assert store2.active()[0]["label"] == "タイマー"
