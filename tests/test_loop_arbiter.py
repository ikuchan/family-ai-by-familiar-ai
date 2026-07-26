"""段階2：軽量LLM 調停の3分岐。

反復の頭で軽量LLM に会話の重さを自己判断させ、(a)軽量で閉じる／(b)フルを起こす／(c)定型
の3つへ振り分ける（`I内部設計根拠` 段4）。閾値は作らず、調整はプロンプトで行う。
実測では1ターン 10.5 秒のうち LLM が 10.2 秒で、`recall` を投げるだけの反復にもフルLLM を
使っていた。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from familiar_agent.loop.arbiter import ARBITER_PROMPT, Decision, arbitrate


def _backend(reply: str):
    b = AsyncMock()
    b.complete = AsyncMock(return_value=reply)
    return b


def _call(reply: str, timeout: float = 2.0) -> Decision:
    return asyncio.run(
        arbitrate(_backend(reply), utterance="こんにちは", workspace_ctx="[想起]…",
                  timeout=timeout)
    )


def test_light_branch_carries_the_reply():
    d = _call('{"branch":"light","text":"やあ！元気？"}')
    assert d.branch == "light"
    assert d.text == "やあ！元気？"


def test_full_branch_carries_the_effort():
    d = _call('{"branch":"full","effort":"medium"}')
    assert d.branch == "full"
    assert d.effort == "medium"


def test_action_branch_carries_the_query():
    d = _call('{"branch":"action","query":"昨日の天気"}')
    assert d.branch == "action"
    assert d.query == "昨日の天気"


def test_action_branch_can_carry_a_filler_and_a_tool_name():
    # つなぎの発話は軽量LLM に出させる（フルLLM を経由すると 2.9 秒かかるところが 0.7 秒）。
    # どの動作で調べるかも軽量LLM が選ぶ（記憶を探すのと外を調べるのは別）。
    d = _call('{"branch":"action","action":"search_deferred",'
              '"query":"今日の天気","text":"調べてみるね"}')
    assert d.branch == "action"
    assert d.action == "search_deferred"
    assert d.text == "調べてみるね"


def test_action_defaults_to_recall_when_no_tool_is_named():
    d = _call('{"branch":"action","query":"昨日の天気"}')
    assert d.action == "recall"


def test_unparsable_reply_falls_back_to_full():
    # 判定できないときは今までと同じ挙動（フル・effort=high）へ倒す＝退行しない。
    d = _call("よくわからない返事")
    assert d.branch == "full"
    assert d.effort == "high"


def test_timeout_falls_back_to_full():
    async def slow(*_a, **_k):
        await asyncio.sleep(1.0)
        return '{"branch":"light","text":"間に合わない"}'

    b = AsyncMock()
    b.complete = AsyncMock(side_effect=slow)
    d = asyncio.run(arbitrate(b, utterance="x", workspace_ctx="", timeout=0.05))
    assert d.branch == "full"
    assert d.effort == "high"


def _prompt_of(backend) -> str:
    """軽量LLM に実際に渡った文面。"""
    return backend.complete.call_args.args[0]


def test_arbiter_speaks_as_the_persona():
    # 発話の出口は2つ（軽量LLM のつなぎ・light／フルLLM の答え）。軽量側にだけ人格が
    # 渡っていないと、同じ人格が2つの口で違う口調で喋る（実機で「調べてくるね！」と
    # 「調べてみますね。」が混ざった）。
    b = _backend('{"branch":"light","text":"やあ"}')
    asyncio.run(arbitrate(b, utterance="こんにちは", workspace_ctx="",
                          self_understanding="名前： パジュ\n一人称：ぼく"))
    assert "パジュ" in _prompt_of(b) and "ぼく" in _prompt_of(b)


def test_arbiter_judges_sufficiency_not_mere_arrival():
    # 「結果が届いたか」ではなく「答えるに足るか」で分ける。足りなければ別の角度で調べ直す。
    assert "[調査中]" not in ARBITER_PROMPT      # 廃止した合成ラベル＝死んだ指示
    assert "足る" in ARBITER_PROMPT


def test_arbiter_is_told_when_no_more_looking_up_is_possible():
    # 上限では action を選ばせない。いまは選ばせてコード側が捨てており、その反復の判断が
    # まるごと無駄になる。
    b = _backend('{"branch":"full"}')
    asyncio.run(arbitrate(b, utterance="?", workspace_ctx="", capped=True))
    assert "これ以上は調べられない" in _prompt_of(b)
    b2 = _backend('{"branch":"full"}')
    asyncio.run(arbitrate(b2, utterance="?", workspace_ctx="", capped=False))
    assert "これ以上は調べられない" not in _prompt_of(b2)


def test_arbiter_gets_the_same_grounding_as_the_full_llm():
    # 発話の出口は2つ。片方にだけ文脈を渡すと、症状が出るたび1つずつ足すことになる
    # （人格を足した翌日、14時39分に「こんばんは」と言った＝日時が無かった）。
    b = _backend('{"branch":"light","text":"やあ"}')
    asyncio.run(arbitrate(b, utterance="こんにちは", workspace_ctx="",
                          self_understanding="名前： パジュ\n## 私にできること\n- 記憶を探せる",
                          family_md="たいき：家族の長男",
                          present_ctx='(present :speaker "たいき")',
                          now_ctx='(now :datetime "2026-07-26 14:39")'))
    p = _prompt_of(b)
    for needle in ("パジュ", "記憶を探せる", "たいき：家族の長男",
                   ':speaker "たいき"', "14:39"):
        assert needle in p


def test_filler_examples_do_not_fix_the_register():
    # つなぎの見本が「調べてみるね」だと、その口調が相手に合わせる規則より近くにあり、
    # パパ（大人＝ですます）にタメ口で「調べてくるね！」と返した（実機で観測）。
    assert "調べてみるね" not in ARBITER_PROMPT


def test_full_branch_also_writes_a_filler_that_avoids_committing_to_content():
    # full のつなぎは答えの前に置かれるので、中身を先取りすると本応答と食い違う。
    assert "内容に触れない" in ARBITER_PROMPT


def test_second_filler_is_asked_to_continue_not_restart():
    # 実機で「調べてみるね」に相当する前置きが5回続いた。つなぎを止めるのではなく、
    # 二言目以降を「まだ考えている最中だと伝わるだけの短い言葉」にさせる。軽量LLM と
    # フルLLM が交互に喋ると、聞いている側には別々の人格が居るように聞こえる。
    assert "その続きとして書く" in ARBITER_PROMPT
    assert "二言目以降" in ARBITER_PROMPT
    assert "同じ人が続けて言っている" in ARBITER_PROMPT


def test_prompt_holds_no_quotable_sample_utterances():
    # カギ括弧で括った「そのまま言える文」を置くと、指示より強く働いて写される。
    # 実機で2度起きた：「調べてみるね」がタメ口を固定し、「もう少しかかりそう」が
    # 「それだけ？」への答えとしてそのまま出た。書き方の説明は残し、見本だけ置かない。
    import re

    for sample in re.findall(r"「([^」]*)」", ARBITER_PROMPT):
        assert sample.startswith("〜") or len(sample) <= 3, f"見本が残っている: {sample}"
