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

from familiar_agent.loop.arbiter import ARBITER_PROMPT, _parse


def _d(reply: str):
    return _parse(reply, can_see=False, origin="発話")


# ── full／action は filler が発話になる ──────────────────────────────────


def test_a_full_turn_speaks_the_filler():
    got = _d(
        '{"branch": "full", "effort": "medium", "filler": "こんにちは。", "text": "用件ぜんぶ", "trash": "も"}'
    )
    assert got is not None
    assert got.text == "こんにちは。", "つなぎは filler から取る"


def test_an_action_turn_speaks_the_filler():
    got = _d(
        '{"branch": "action", "action": "recall", "query": "本", "filler": "調べてみますね。", "text": "捨てる"}'
    )
    assert got is not None
    assert got.text == "調べてみますね。"


def test_the_trash_never_becomes_speech():
    got = _d(
        '{"branch": "full", "effort": "high", "filler": "少し待ってください。", "trash": "メモを更新した内容、覚えました"}'
    )
    assert got is not None
    assert "メモ" not in got.text


def test_a_full_turn_without_a_filler_says_nothing():
    """空なら黙る（いまと同じ扱い）。text を拾い直さない。"""
    got = _d('{"branch": "full", "effort": "medium", "text": "これは使わない"}')
    assert got is not None
    assert got.text == ""


# ── light はこれまでどおり text が返事 ───────────────────────────────────


def test_a_light_turn_still_speaks_the_text():
    got = _d('{"branch": "light", "text": "おかえりなさい。"}')
    assert got is not None
    assert got.text == "おかえりなさい。"


def test_a_light_turn_ignores_the_filler():
    got = _d('{"branch": "light", "text": "おかえりなさい。", "filler": "まちがい"}')
    assert got is not None
    assert got.text == "おかえりなさい。"


# ── プロンプトが 2 つの欄を言う ──────────────────────────────────────────


def test_the_prompt_names_both_boxes():
    assert '"filler"' in ARBITER_PROMPT
    assert '"trash"' in ARBITER_PROMPT


def test_the_filler_instruction_no_longer_points_at_text():
    from familiar_agent.loop.arbiter import _BRANCHES_REPLY

    assert "待ってもらうための短い一言を text に書く" not in _BRANCHES_REPLY
    assert "filler" in _BRANCHES_REPLY
    assert "trash" in _BRANCHES_REPLY


def test_an_action_without_an_action_falls_back_to_light_with_the_filler():
    """動作が無く言葉だけなら light として扱う（実機 23:14 の既存の守り）。

    その言葉は `filler` に書かれている。**`text` を拾い直さない**——拾うと、捨てるつもりで
    書いたものが声になる。
    """
    got = _d('{"branch": "action", "filler": "鳴らしていい？", "text": "捨てるつもりの長い話"}')
    assert got is not None
    assert got.branch == "light"
    assert got.text == "鳴らしていい？"
