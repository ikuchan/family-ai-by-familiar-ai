"""段階2：軽量LLM 調停の3分岐。

反復の頭で軽量LLM に会話の重さを自己判断させ、(a)軽量で閉じる／(b)フルを起こす／(c)定型
の3つへ振り分ける（`I内部設計根拠` 段4）。閾値は作らず、調整はプロンプトで行う。
実測では1ターン 10.5 秒のうち LLM が 10.2 秒で、`recall` を投げるだけの反復にもフルLLM を
使っていた。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.arbiter import (
    _FIELD_TEXT,
    JUDGE_GUIDE,
    SEE_GUIDE,
    WRITER_PROMPT,
    Arbiter,
    ArbiterInput,
    Decision,
)
from tests._arbiter_compat import arbitrate


def _backend(reply: str):
    b = AsyncMock()
    b.complete = AsyncMock(return_value=reply)
    return b


def _call(reply: str, timeout: float = 2.0) -> Decision:
    return asyncio.run(
        arbitrate(_backend(reply), utterance="こんにちは", workspace_ctx="[想起]…", timeout=timeout)
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
    d = _call(
        '{"branch":"action","action":"search_deferred","query":"今日の天気","filler":"調べてみるね"}'
    )
    assert d.branch == "action"
    assert d.action == "search_deferred"
    assert d.text == "調べてみるね"


def test_action_defaults_to_recall_when_no_tool_is_named():
    d = _call('{"branch":"action","query":"昨日の天気"}')
    assert d.action == "recall"


def test_unparsable_reply_falls_back_to_full():
    # 判定できないときはフルへ倒す。effort は既定の low（2026-09-12・課題5 G 章）。
    d = _call("よくわからない返事")
    assert d.branch == "full"
    assert d.effort == "low"


def test_timeout_falls_back_to_full():
    async def slow(*_a, **_k):
        await asyncio.sleep(1.0)
        return '{"branch":"light","text":"間に合わない"}'

    b = AsyncMock()
    b.complete = AsyncMock(side_effect=slow)
    d = asyncio.run(arbitrate(b, utterance="x", workspace_ctx="", timeout=0.05))
    assert d.branch == "full"
    assert d.effort == "low"  # 倒れたときも low（課題5 G 章）


def _prompt_of(backend) -> str:
    """軽量LLM に実際に渡った文面。"""
    return backend.complete.call_args.args[0]


def _jev_state(**kw) -> str:
    """Jev に送る文（分岐を決める側・出-au 段 5-7d）。"""
    return Arbiter(jev=None, writer=None)._state(ArbiterInput(**{"workspace_ctx": "", **kw}))


def _jev_actions(**kw) -> dict:
    """Jev に選ばせる動作の選択肢。"""
    qs = Arbiter(jev=None, writer=None)._questions(ArbiterInput(**{"workspace_ctx": "", **kw}))
    return qs["action"]["criteria"]


#: 調停に渡る指示の文（Jev の目安と、軽量LLM の文章の口）。
_ALL_TEXT = JUDGE_GUIDE + SEE_GUIDE + WRITER_PROMPT + "".join(_FIELD_TEXT.values())


def test_arbiter_speaks_as_the_persona():
    # 発話の出口は2つ（軽量LLM のつなぎ・light／フルLLM の答え）。軽量側にだけ人格が
    # 渡っていないと、同じ人格が2つの口で違う口調で喋る（実機で「調べてくるね！」と
    # 「調べてみますね。」が混ざった）。
    b = _backend('{"branch":"light","text":"やあ"}')
    asyncio.run(
        arbitrate(
            b,
            utterance="こんにちは",
            workspace_ctx="",
            self_understanding="名前： パジュ\n一人称：ぼく",
        )
    )
    # 人格はシステム文で渡す（出-e-に）。**片方の口にだけ渡さない**という性質は同じ。
    system = b.complete.await_args.kwargs["system"]
    assert "パジュ" in system and "ぼく" in system


def test_arbiter_judges_sufficiency_not_mere_arrival():
    # 「結果が届いたか」ではなく「答えるに足るか」で分ける。足りなければ別の角度で調べ直す。
    assert "[調査中]" not in _ALL_TEXT  # 廃止した合成ラベル＝死んだ指示
    assert "足る" in JUDGE_GUIDE


def test_arbiter_is_told_when_no_more_looking_up_is_possible():
    # 上限では action を選ばせない。いまは選ばせてコード側が捨てており、その反復の判断が
    # まるごと無駄になる。
    # 分岐を決めるのは Jev なので、Jev に送る文に載り、action は選択肢から外れる（出-au 段 5-7d）。
    assert "これ以上は調べられない" in _jev_state(utterance="?", capped=True)
    assert "これ以上は調べられない" not in _jev_state(utterance="?", capped=False)
    qs = Arbiter(jev=None, writer=None)._questions(
        ArbiterInput(utterance="?", workspace_ctx="", capped=True)
    )
    assert "action" not in qs["branch"]["criteria"]


def test_arbiter_gets_the_same_grounding_as_the_full_llm():
    # 発話の出口は2つ。片方にだけ文脈を渡すと、症状が出るたび1つずつ足すことになる
    # （人格を足した翌日、14時39分に「こんばんは」と言った＝日時が無かった）。
    b = _backend('{"branch":"light","text":"やあ"}')
    asyncio.run(
        arbitrate(
            b,
            utterance="こんにちは",
            workspace_ctx="",
            self_understanding="名前： パジュ\n## 私にできること\n- 記憶を探せる",
            family_md="たいき：家族の長男",
            present_ctx='(present :speaker "たいき")',
            now_ctx='(now :datetime "2026-07-26 14:39")',
        )
    )
    # 文脈は2箇所へ分かれた（出-e-に）。**合わせて見る**——身元はシステム文、
    # いま誰が居るかと時刻はプロンプトである。片方にだけ渡す形へ戻っていないこと。
    system = b.complete.await_args.kwargs["system"]
    prompt = _prompt_of(b)
    for needle in ("パジュ", "記憶を探せる", "たいき：家族の長男"):
        assert needle in system, needle
    for needle in (':speaker "たいき"', "14:39"):
        assert needle in prompt, needle


def test_filler_examples_do_not_fix_the_register():
    # つなぎの見本が「調べてみるね」だと、その口調が相手に合わせる規則より近くにあり、
    # パパ（大人＝ですます）にタメ口で「調べてくるね！」と返した（実機で観測）。
    assert "調べてみるね" not in _ALL_TEXT


def _rendered_reply_prompt() -> str:
    """人の発話が起点のとき、軽量LLM に実際に渡る文面（分岐の説明は起点で差し替わる・情-e）。"""
    b = MagicMock()
    # full でも深さが low なら軽量LLM は呼ばれない（出-au 段 5-7c）。つなぎを書かせる medium で見る。
    b.complete = AsyncMock(return_value='{"branch": "full", "effort": "medium", "filler": "うん"}')
    asyncio.run(arbitrate(b, utterance="x", workspace_ctx="", origin="発話"))
    return b.complete.call_args.args[0]


def test_full_branch_also_writes_a_filler_that_avoids_committing_to_content():
    # full のつなぎは答えの前に置かれるので、中身を先取りすると本応答と食い違う。
    assert "内容に触れない" in _rendered_reply_prompt()


def test_second_filler_is_asked_to_continue_not_restart():
    # 実機で「調べてみるね」に相当する前置きが5回続いた。つなぎを止めるのではなく、
    # 二言目以降を「まだ考えている最中だと伝わるだけの短い言葉」にさせる。軽量LLM と
    # フルLLM が交互に喋ると、聞いている側には別々の人格が居るように聞こえる。
    assert "その続きとして書く" in WRITER_PROMPT
    assert "二言目以降" in WRITER_PROMPT
    assert "同じ人が続けて言っている" in WRITER_PROMPT


def test_prompt_holds_no_quotable_sample_utterances():
    # カギ括弧で括った「そのまま言える文」を置くと、指示より強く働いて写される。
    # 実機で2度起きた：「調べてみるね」がタメ口を固定し、「もう少しかかりそう」が
    # 「それだけ？」への答えとしてそのまま出た。書き方の説明は残し、見本だけ置かない。
    import re

    questions = Arbiter(jev=None, writer=None)._questions(
        ArbiterInput(utterance="x", workspace_ctx="", silenced=True)
    )
    text = _ALL_TEXT + _jev_state(utterance="x") + str(questions)
    for sample in re.findall(r"「([^」]*)」", text):
        assert sample.startswith("〜") or len(sample) <= 3, f"見本が残っている: {sample}"


def test_tone_rule_sits_next_to_where_the_filler_is_asked_for():
    # 口調の指示は分岐の説明から離れた位置にあり、短いつなぎのときだけ守られなかった
    # （実機で、本応答はですますなのに待ってもらう一言だけタメ口）。文章を書けと言う
    # 場所（決まったこと）の直後へ置く。キャッシュ境界（毎分変わる [いま]）より手前。
    prompt = _rendered_reply_prompt()
    tone = prompt.index("text と filler の口調は")
    assert prompt.index("次にすることはもう決まっている") < tone
    assert tone < prompt.index("[いま]")
    assert "短い一言でも同じ" in prompt


def test_the_arbiter_has_a_way_out_when_nothing_more_can_be_found():
    """行き止まりの出口を持たせる。

    分かれ目が「材料が無い→調べる／答えきれない→調べ直す／足る→答える」の3つだけだと、
    材料が足りない限り**必ず再検索へ向かう**。実機で、同じ `recall` を4反復続けて投げた
    （語は MD5 まで一致・結果も毎回同じ）。一覧は調停に届いていた（機構としては確認済み）
    ので、足りなかったのは「これ以上は分からない」と言う選択肢だった。
    """
    assert "分からないと伝える" in JUDGE_GUIDE
    assert "すでに調べた語と同じ語では投げない" in JUDGE_GUIDE


def test_an_affect_origin_can_choose_a_synchronous_mcp_tool():
    """情動が起点の求めでも、繋がっている MCP の同期の道具（家の決まり・予定）が候補に載る（知-g-は）。

    09-13 に「後回し」としたが、知-j（動作の表）と出-p（候補文）で通っていた。証拠として置く。
    """
    b = _backend('{"branch":"action","action":"house_rules"}')
    d = asyncio.run(
        arbitrate(
            b,
            utterance="[内的な促し:SEEKING] 探索したい",
            workspace_ctx="",
            origin="情動",
            extra_actions=("house_rules", "family_schedule"),
        )
    )
    # 動作は Jev の選択肢（出-au 段 5-7d）。
    actions = _jev_actions(
        utterance="x", origin="情動", extra_actions=("house_rules", "family_schedule")
    )
    assert {"recall", "search_deferred", "house_rules", "family_schedule"} <= set(actions)
    b.complete.assert_not_awaited()  # 行き先の決まった道具は、書くものが無いので軽量LLM を呼ばない
    assert d.branch == "action" and d.action == "house_rules" and d.query == "家の決まりを見る"
    assert d.text == ""  # 自発の行動に断りは要らない（情-e）


def test_a_request_that_needs_a_tool_is_never_answered_lightly():
    """道具が要る頼み（タイマー・アラーム・測る・止める）は light で「できました」と言わない。

    実機（2026-09-15 22:47）で「３分のタイマーをかけて」に調停が light を選び、道具を呼ばず
    「タイマーをセットしました」と言った。掛かっていない。機械で full へ倒す。
    """
    from familiar_agent.loop.arbiter import needs_tools

    for text in (
        "３分のタイマーをかけて",
        "7時に起こして",
        "今から測って",
        "タイマー止めて",
        "アラーム掛けといて",
        "5分後に教えて",
    ):
        assert needs_tools(text), text
        d = asyncio.run(
            arbitrate(
                _backend('{"branch":"light","text":"セットしました"}'),
                utterance=text,
                workspace_ctx="",
            )
        )
        assert d.branch == "full" and d.effort == "low", text
    assert not needs_tools("こんばんは")
    assert not needs_tools("時間ある？")
    d = asyncio.run(
        arbitrate(
            _backend('{"branch":"light","text":"やあ"}'), utterance="こんばんは", workspace_ctx=""
        )
    )
    assert d.branch == "light"
    # Jev の選択肢の説明にも書いてある（機械の守りは最後の砦）。
    qs = Arbiter(jev=None, writer=None)._questions(ArbiterInput(utterance="x", workspace_ctx=""))
    assert "道具が要る" in qs["branch"]["criteria"]["light"]


def test_the_silence_request_survives_the_fall_to_full():
    d = asyncio.run(
        arbitrate(
            _backend('{"branch":"light","text":"わかった","silence_minutes":30}'),
            utterance="話すの止めて",
            workspace_ctx="",
        )
    )
    assert d.branch == "full" and d.silence_minutes == 30


def test_coming_back_from_a_look_of_its_own_is_not_framed_as_answering_someone():
    """自分から見に行った帰り（情動）は、返事の場面の書き方を渡さない。

    実機（2026-09-16 09:16 と 09:51）で、SAFETY／SEEKING の促しで見に行った帰りの調停が
    「はい、静かにしていますね」と、誰にも聞かれていないのに返事の体裁の一言を作った。自分の帰りには
    「いつも通りなら黙る」を渡す。返事の場面は従来どおり（対で確認）。写真は調停に渡らない（出-au 段 5-7a）。
    """
    b = _backend('{"branch":"light","text":""}')
    asyncio.run(
        arbitrate(
            b,
            utterance="[内的な促し:SAFETY] 確かめたい気持ちが湧いている。見回る。",
            workspace_ctx="",
            origin="情動",
            can_see=True,
        )
    )
    own = _prompt_of(b)
    assert "いつも通りなら" in own and "黙る" in own
    assert "返事や約束の形" in own

    b = _backend('{"branch":"light","text":"椅子と机が見えるよ"}')
    asyncio.run(arbitrate(b, utterance="何が見える？", workspace_ctx="", can_see=True))
    assert "いつも通りなら" not in _prompt_of(b)


def test_a_self_driven_turn_treats_the_recent_exchange_as_already_over():
    """自発の求めでは、直近のやりとりは済んだこととして渡す（出-q）。

    実機（2026-09-15 15:56・16:03）で `seeking`／`safety` の自発ターンが、済んだ入室に
    もう一度「パパ、おかえりなさい」と言った。09-16 09:16 の W を見ると、自発の求めに
    「[入室] 誰か が来た」と自分の挨拶が『直近のやりとり』として載っており、`_LEAD_SELF` には
    済んだ出来事に反応しないという材料が無かった。返事の場面（発話が起点）は従来どおり。
    """
    b = _backend('{"branch":"light","text":""}')
    asyncio.run(
        arbitrate(
            b,
            utterance="[内的な促し:SEEKING] 探索したい気持ちが湧いている。",
            workspace_ctx="[直近のやりとり]\n- 09:16 きっかけ：[入室] 誰か が来た\n- 09:16 わたし：あ、パパだ。おはようございます",
            origin="情動",
        )
    )
    own = _prompt_of(b)
    assert "済んだこと" in own
    assert "改めて反応しない" in own
    assert "切り上げていたら" in own

    b = _backend('{"branch":"light","text":"おはよう"}')
    asyncio.run(arbitrate(b, utterance="おはよう", workspace_ctx=""))
    assert "済んだこと" not in _prompt_of(b)


# ── 首を向ける（`look`）を調停の候補に（2026-09-16 実機 11:34）────────────────
#
# 「右見れる?」「もっと右を見て」に調停は `see`（いまの向きで撮る）しか選べず、正面のまま
# 「右側はタンスと椅子が見えていますよ」と答えた。`look` は主LLM の道具にしか無く、
# `needs_tools` にも首振りの語が無いので、そこへも行かない。タイマーと同じ軽量の経路で
# 首を回す。


def _seeing(reply: str):
    b = _backend(reply)
    return b, asyncio.run(arbitrate(b, utterance="右見れる？", workspace_ctx="", can_see=True))


def test_the_arbiter_can_turn_the_head():
    b, d = _seeing('{"branch":"action","action":"look","tool_input":{"direction":"右"}}')
    assert {"recall", "search_deferred", "see", "look"} <= set(
        _jev_actions(utterance="右見れる？", can_see=True)
    )
    assert d.branch == "action" and d.action == "look"
    assert d.tool_input == {"direction": "右"}
    assert d.query  # (c) 分岐は query が空だと full へ落ちる
    assert d.text == ""  # 首を回すだけなので断らない


def test_a_direction_or_pose_written_in_query_becomes_the_input():
    # 文章の口は look に tool_input だけを頼む。query に書いてきたときの読み替えは組み立て（assemble）の守り。
    from tests._arbiter_compat import _parse

    d = _parse('{"branch":"action","action":"look","query":"右"}', can_see=True)
    assert d.tool_input == {"direction": "右"}
    d = _parse('{"branch":"action","action":"look","query":"窓"}', can_see=True)
    assert d.tool_input == {"pose": "窓"}


def test_look_is_not_offered_without_a_camera():
    assert "look" not in _jev_actions(utterance="右見れる？", can_see=False)
    # 選択肢に無い動作が返ってきても（偽の Jev）、首は回さない。
    b = _backend('{"branch":"action","action":"look","tool_input":{"direction":"右"}}')
    d = asyncio.run(arbitrate(b, utterance="右見れる？", workspace_ctx="", can_see=False))
    assert d.action != "look"


def test_the_label_of_a_look_names_the_direction():
    from familiar_agent.loop.event_loop import _query_label

    assert _query_label("look", {"direction": "右"}) == "右を見に行く"
    assert _query_label("look", {"pose": "窓"}) == "窓を見に行く"


def test_the_see_guide_tells_the_arbiter_the_head_is_already_turned():
    """`look` の帰りで `look` を選び直さない（実機 15:11・3 回選び直して 18 秒）。目安は Jev に送る文にある。"""
    assert "首はもう向いている" in SEE_GUIDE and "もう一度 look は選ばない" in SEE_GUIDE
    assert "首はもう向いている" in _jev_state(utterance="右向いて", can_see=True)
    assert "首はもう向いている" not in _jev_state(utterance="右向いて", can_see=False)
