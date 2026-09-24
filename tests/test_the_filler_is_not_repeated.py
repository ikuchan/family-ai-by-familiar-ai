"""つなぎの直後に、同じ挨拶を繰り返さない（出-aj #4・2026-09-24）。

実機 15:49、つなぎで「こんにちは。」と言った 6 秒後、本応答が
「**こんにちは！**さっきタイマーが鳴っていたので、止めておきましたよ」と始めた。
相手はもう聞いている。

**言葉では止まらなかった。** 9 通り試して、最良が 8 回中 2 回である。

| 試した言い方 | 繰り返さなかった |
|---|---|
| W の「すでに伝えた一言」に置く／可変部の行／`[返事]` の指示／両方／`[人の言葉]` に添える／静的核の `voice`／W を時間順に 1 本化／枠を分ける | いずれも 0 |
| 規則で**探す場所を名指しする** | 2/8 |

基準（いまのまま）は **8/8 で繰り返す**——W の作りを変えた後（出-ar・出-ap）も直っていない。

**どちらも自分が言った言葉である。** 意味を判断する必要はなく、2 秒差で同じことを 2 回
言っているのを数えて落とせばよい。調停の分岐（挨拶に `full` を選ばない）は本人が退けた
——「**挨拶かどうか分からない。順序が逆**」。先に挨拶を見分ける必要があるのは、そちらの案である。

**落とすのは冒頭だけ。** 途中の挨拶（`タイマー止めましたよ。こんにちは、どなたですか？`）は
残る。そこを落とすには意味の判断が要るので、機械ではやらない。
"""

from __future__ import annotations

from familiar_agent.core.filler_echo import drop_echo


# ── 実機の文字列 ──────────────────────────────────────────────────────────


def test_the_greeting_said_a_moment_ago_is_dropped():
    got = drop_echo(
        "こんにちは！さっきタイマーが鳴っていたので、止めておきましたよ。", ["こんにちは。"]
    )
    assert got == "さっきタイマーが鳴っていたので、止めておきましたよ。"


def test_the_punctuation_does_not_have_to_match():
    """つなぎは `。`、本応答は `！`。**同じ言葉かどうかだけを見る。**"""
    assert drop_echo("こんにちは、どなたですか？", ["こんにちは！"]) == "どなたですか？"


def test_a_greeting_no_table_would_know_is_dropped_too():
    """挨拶の一覧を持たない。**つなぎに現れたかどうか**だけで決める。"""
    got = drop_echo("おかえりなさい。荷物、重そうでしたね。", ["おかえりなさい、パパ。"])
    assert got == "荷物、重そうでしたね。"


# ── 落としてはいけないもの ────────────────────────────────────────────────


def test_a_reply_that_does_not_repeat_is_untouched():
    text = "タイマーは止めておきましたよ。ところで、どなたですか？"
    assert drop_echo(text, ["こんにちは。"]) == text


def test_nothing_happens_without_a_filler():
    text = "こんにちは！さっきタイマーが鳴っていました。"
    assert drop_echo(text, []) == text
    assert drop_echo(text, None) == text


def test_a_greeting_in_the_middle_stays():
    """**冒頭だけ。** 途中を落とすには意味の判断が要る。"""
    text = "タイマー止めましたよ。こんにちは、どなたですか？"
    assert drop_echo(text, ["こんにちは。"]) == text


def test_a_one_character_opening_is_left_alone():
    """「あ」「え」で誤爆させない。"""
    assert drop_echo("あ、タイマー止めましたよ。", ["あ、そうだ。"]) == "あ、タイマー止めましたよ。"


def test_a_reply_that_is_only_the_echo_is_kept():
    """落とすと何も残らないなら触らない。**黙らせない。**"""
    assert drop_echo("こんにちは。", ["こんにちは。"]) == "こんにちは。"


def test_only_the_first_piece_is_dropped():
    """連鎖させない。1 回落としたら、そこで止める。"""
    got = drop_echo("こんにちは。こんにちは。元気ですか？", ["こんにちは。"])
    assert got == "こんにちは。元気ですか？"


def test_several_fillers_are_all_compared():
    """つなぎは 1 回とは限らない（調べもの待ちでは繰り返し出る）。"""
    got = drop_echo(
        "おまたせしました。答えが出ましたよ。", ["少し待ってね。", "おまたせしました。"]
    )
    assert got == "答えが出ましたよ。"


def test_an_empty_reply_is_fine():
    assert drop_echo("", ["こんにちは。"]) == ""


# ── 本応答の経路を通る ────────────────────────────────────────────────────


def test_the_loop_drops_the_echo_before_the_check():
    """**検査には出す文を見せる。** 落とす前の文で整合を見ても意味がない。"""
    import inspect

    from familiar_agent.loop.event_loop import InformationProcessing

    src = inspect.getsource(InformationProcessing._act_on_decision)
    assert "drop_echo" in src
    assert src.index("drop_echo") < src.index("_coherence_violation")
