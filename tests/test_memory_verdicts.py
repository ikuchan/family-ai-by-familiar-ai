"""フルLLM が「想起した記憶をどう扱ったか」を申告する（課題5 E節 段2）。

設計は「**フルLLM が参照した MI だけ**再評価（上げ下げ）＋freshness 更新」と定めている。
その更新契機がこの申告である。想起しただけで更新すると、一度上がった記録が自分を押し上げ
続ける（実機で 47日前の挨拶が t=1.000 で居座り、5秒前の自分の発話を押し出した）。

判定は4つ。**W に出た記憶すべて**について返させる。

- important（大事）　　 n += 1 ＋ 時間の起点を更新
- useless（不要）　　　 n -= 1 ＋ 時間の起点を更新
- referred（参照）　　　時間の起点だけ更新
- unused（使わなかった）何もしない

**照合できたものだけ適用する。** 落とされた分を「使わなかった」と決めつけると、申告漏れと
本当に使わなかったことを混同する。
"""

from __future__ import annotations

from familiar_agent.loop import workspace
from familiar_agent.loop.request import Request

from unittest.mock import MagicMock

from familiar_agent.backends import ToolCall
from familiar_agent.loop.prompt import EVENT_SYSTEM_PROMPT
from tests.test_event_loop import _agent, _run, _turn


def test_the_say_tool_accepts_verdicts():
    from familiar_agent.tools.tts import TTSTool

    schema = TTSTool.get_tool_definitions(MagicMock())[0]["input_schema"]
    verdicts = schema["properties"]["memory_verdicts"]
    assert verdicts["items"]["properties"]["verdict"]["enum"] == [
        "important",
        "useless",
        "referred",
        "unused",
    ]
    assert "memory_verdicts" in schema["required"]  # 必須（出-h-い）


def test_the_schema_tells_how_to_choose_each_verdict():
    """**4つの判定に、別々の引き金を与える。**

    引き金が無いと、無難な `referred` が全件に並び、W の記憶が一斉に若返る（47日前の
    挨拶が居座った形）。逆に「迷うなら unused」だけだと `important` が1件も出ず、
    `groundedness_n` が増えない。**両側へ倒れないよう、条件で分ける。**
    """
    from familiar_agent.tools.tts import TTSTool

    schema = TTSTool.get_tool_definitions(MagicMock())[0]["input_schema"]
    desc = schema["properties"]["memory_verdicts"]["description"]
    for verdict in ("important", "useless", "referred", "unused"):
        assert f"`{verdict}`" in desc, f"{verdict} の引き金が書かれていない"
    # `important` と `referred` を分ける軸は「この反復だけか、この先も効くか」。
    assert "beyond this turn" in desc and "only for this turn" in desc
    # 欠けさせない指示。
    assert "every id" in desc


def test_the_schema_is_the_same_every_turn():
    """**道具の定義は反復をまたいで変わらない。**

    道具は安定部と同じキャッシュ範囲にある（出-i）。W の id を `enum` へ入れると、想起が
    変わるたび定義が変わり、毎ターン書き直しになる（1000ターン 366円 → 738円）。
    実測では `enum` の有無で申告の成績は変わらなかったので、入れない。
    """
    from familiar_agent.tools.tts import TTSTool

    a = TTSTool.get_tool_definitions(MagicMock())[0]
    b = TTSTool.get_tool_definitions(MagicMock())[0]
    assert a == b
    id_schema = a["input_schema"]["properties"]["memory_verdicts"]["items"]["properties"]["id"]
    assert "enum" not in id_schema


def test_the_prompt_asks_for_every_recalled_memory():
    assert "memory_verdicts" in EVENT_SYSTEM_PROMPT
    assert "important" in EVENT_SYSTEM_PROMPT and "unused" in EVENT_SYSTEM_PROMPT


def test_verdicts_are_matched_through_the_index_not_by_prefix_guessing():
    # 写し間違いは一致せず、黙って別の記憶へ適用されない。
    mem = MagicMock()
    workspace.apply_memory_verdicts(
        mem,
        [
            {"id": "aaaaaaaaaaaa", "verdict": "important"},
            {"id": "zzzzzzzzzzzz", "verdict": "useless"},  # W に無い id
        ],
        {"aaaaaaaaaaaa": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"},
    )
    applied = mem.apply_verdicts.call_args.args[0]
    assert applied == {"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa": "important"}


def test_verdicts_land_on_the_memory_that_the_recall_came_from():
    """**申告は、想起に使った記憶オブジェクトへ当てる**（出-h-ろ ③）。

    想起は `agent._active_memory()`＝**話者の面**を通る（`event_loop.py`）。申告が
    `agent._memory`（基底＝`__self__` の面）へ行くと、話者が同定されている場面で
    `UPDATE ... WHERE person_id = '__self__'` が0行を返し、**申告が効かない**。
    `situated_memories` は人ごとなので、引いた面と書く面は同じでなければならない。
    """
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    speaker_mem = MagicMock()  # 話者の面（`_active_memory()` が返すもの）
    workspace.apply_memory_verdicts(
        speaker_mem,
        [{"id": "aaaaaaaaaaaa", "verdict": "important"}],
        {"aaaaaaaaaaaa": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"},
    )
    speaker_mem.apply_verdicts.assert_called_once()
    a._memory.apply_verdicts.assert_not_called()  # 基底の面へは行かない


def test_nothing_is_applied_when_the_workspace_had_no_memories():
    mem = MagicMock()
    # **対応表に既定値は無い**（に-5-に-2）。空を渡せば何も当たらない。
    workspace.apply_memory_verdicts(mem, [{"id": "aaaaaaaaaaaa", "verdict": "important"}], {})
    mem.apply_verdicts.assert_not_called()


def test_the_workspace_prints_twelve_digit_ids():
    # 8桁だと記録が10万件規模でほぼ確実に衝突する。
    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "はい"})])])
    _run(a, utterance="おはよう")
    _text, id_map = workspace.compose(
        a._active_memory(), [{"memory_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"}], Request()
    )
    assert list(id_map) == ["aaaaaaaabbbb"]
