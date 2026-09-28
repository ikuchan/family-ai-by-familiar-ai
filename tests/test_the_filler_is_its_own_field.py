"""つなぎを、返事とは別の欄で受ける（出-aj・2026-09-23）。

実機 15:49、「こんにちは」に対して 144 字のつなぎが出た——「…どなたでしょうか？ あ、それ
から、お伝えしたいことがいくつかあります。メモを更新した内容、覚えました。フーコック旅行
は…」。**本応答は 35 字で混ぜていない。** 混ぜたのはつなぎのほうだった。

同じ `text` という欄を 3 つの用途が共有していたのが根である——`light` では**それが返事**、
`full`／`action` では**つなぎ**。`light` の書き方が既定になり、つなぎも返事として書かれていた。

文言では止まらなかった（3 通り測って 6/6 で混ぜる）。**欄を分け、言いたいことの行き先
（`trash`）を置くと 0/6 になる。** 削って測ると、要るのはこの 2 つだけだった——「空にしない」
と JSON の並び替えは効かない（`trash` が無いと `filler` は 6 回中 5 回が空になる）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.arbiter import Arbiter, ArbiterInput, assemble


def _d(data: dict):
    """Jev の答えと軽量LLM の文章を合わせた辞書を、決定に組み立てる。"""
    return assemble(data, can_see=False, origin="発話")


# ── full／action は filler が発話になる ──────────────────────────────────


def test_a_full_turn_speaks_the_filler():
    got = _d(
        {
            "branch": "full",
            "effort": "medium",
            "filler": "こんにちは。",
            "text": "用件ぜんぶ",
            "trash": "も",
        }
    )
    assert got is not None
    assert got.text == "こんにちは。", "つなぎは filler から取る"


def test_an_action_turn_speaks_the_filler():
    got = _d(
        {
            "branch": "action",
            "action": "recall",
            "query": "本",
            "filler": "調べてみますね。",
            "text": "捨てる",
        }
    )
    assert got is not None
    assert got.text == "調べてみますね。"


def test_the_trash_never_becomes_speech():
    got = _d(
        {
            "branch": "full",
            "effort": "high",
            "filler": "少し待ってください。",
            "trash": "メモを更新した内容、覚えました",
        }
    )
    assert got is not None
    assert "メモ" not in got.text


def test_a_full_turn_without_a_filler_says_nothing():
    """空なら黙る（いまと同じ扱い）。text を拾い直さない。"""
    got = _d({"branch": "full", "effort": "medium", "text": "これは使わない"})
    assert got is not None
    assert got.text == ""


# ── light はこれまでどおり text が返事 ───────────────────────────────────


def test_a_light_turn_still_speaks_the_text():
    got = _d({"branch": "light", "text": "おかえりなさい。"})
    assert got is not None
    assert got.text == "おかえりなさい。"


def test_a_light_turn_ignores_the_filler():
    got = _d({"branch": "light", "text": "おかえりなさい。", "filler": "まちがい"})
    assert got is not None
    assert got.text == "おかえりなさい。"


# ── 文章の口が 2 つの欄を言う（出-au 段 5-7d・軽量LLM の文章の側）───────────


def _asked_for(data: dict) -> str:
    w = MagicMock()
    w.complete = AsyncMock(return_value='{"filler": "うん"}')
    inp = ArbiterInput(utterance="こんにちは", workspace_ctx="（なし）")
    asyncio.run(Arbiter(jev=None, writer=w, timeout=2.0)._write(inp, data))
    return w.complete.await_args.args[0]


def test_a_filler_is_asked_for_with_a_place_to_throw_away():
    """つなぎを書かせるときは、言いたいことの行き先（`trash`）も並べる。無いと用件がつなぎへ流れる。"""
    asked = _asked_for({"branch": "full", "effort": "medium"})
    assert '"filler"' in asked
    assert '"trash"' in asked
    assert '"text"' not in asked


def test_an_action_filler_also_gets_the_place_to_throw_away():
    asked = _asked_for({"branch": "action", "action": "search_deferred"})
    assert '"filler"' in asked and '"trash"' in asked


def test_a_light_reply_has_no_place_to_throw_away():
    """light では text が返事そのもの。捨て場は要らない。"""
    asked = _asked_for({"branch": "light"})
    assert '"trash"' not in asked


def test_an_action_without_an_action_falls_back_to_light_with_the_filler():
    """動作が無く言葉だけなら light として扱う（実機 23:14 の既存の守り）。

    その言葉は `filler` に書かれている。**`text` を拾い直さない**——拾うと、捨てるつもりで
    書いたものが声になる。
    """
    got = _d({"branch": "action", "filler": "鳴らしていい？", "text": "捨てるつもりの長い話"})
    assert got is not None
    assert got.branch == "light"
    assert got.text == "鳴らしていい？"
