"""道具呼び出しを**関数呼び出し風の文**で書いてきたときも拾う（2026-09-15 実機）。

`say(text="パパ、…", <並びの引数>=[{…}, …])` が本文として返り、
ループは「道具なし・542 字」と見て**沈黙**にした（パジュへのメモの初回・20:12）。`<invoke>` の形
（09-13）に加えて、この形も `ToolCall` に直す。当時の並びの引数は記憶の申告で、記-r（2026-10-14）で `say` から
外したので、例はいまの並びの引数（見立て `seen_people`）に替えた。
"""

from __future__ import annotations

from familiar_agent.core.tool_text import tool_calls_from_text

SAY = (
    'say(text="パパ、今日は体調崩して午後お休みしてたんだね。もう大丈夫？キャンプの件、9/17までに決めるやつだね、覚えとくよ。", '
    'seen_people=[{"name":"パパ","confidence":0.8},{"name":"","confidence":0.2}])'
)


def test_a_call_style_say_is_recovered_with_its_arguments():
    calls = tool_calls_from_text(SAY)
    assert len(calls) == 1 and calls[0].name == "say"
    assert (
        calls[0].input["text"].startswith("パパ、今日は体調崩して")
        and "9/17" in calls[0].input["text"]
    )
    assert calls[0].input["seen_people"][0] == {"name": "パパ", "confidence": 0.8}


def test_numbers_bools_and_single_quotes_are_read():
    calls = tool_calls_from_text("set_timer(after_minutes=3, label='パスタ', silent=true)")
    assert calls and calls[0].name == "set_timer"
    assert calls[0].input == {"after_minutes": 3, "label": "パスタ", "silent": True}


def test_plain_prose_is_not_mistaken_for_a_call():
    assert tool_calls_from_text("パパ、おかえりなさい（今日は雨だね）。") == []
    assert tool_calls_from_text("say(") == []
    assert tool_calls_from_text('こんにちは。 say(text="x")') == []  # 先頭でなければ文
    assert tool_calls_from_text("") == []


def test_the_invoke_form_still_works():
    calls = tool_calls_from_text(
        '<invoke name="recall"><parameter name="query">昨日</parameter></invoke>'
    )
    assert calls and calls[0].name == "recall" and calls[0].input == {"query": "昨日"}
