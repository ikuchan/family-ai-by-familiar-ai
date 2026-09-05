"""調停プロンプトの並び：固定のものを先、変わるものを後へ。

プロンプトキャッシュは**前方一致**で効く。変わりうるものが固定のものより前にあると、それが
変わるたびに後ろ全部が作り直しになる。

実機で、調停が 2 秒で返らず時間切れになった（沈黙依頼が読まれないまま `full` へ倒れた）。
渡すものを増やしてきたので、キャッシュが効く形に組み直す。

名前は `[あなたは誰か]`（`ME.md`）に既に入っている。同じものを別枠でもう一度渡さない。
"""

from __future__ import annotations

from familiar_agent.loop.arbiter import ARBITER_PROMPT


def _pos(marker: str) -> int:
    i = ARBITER_PROMPT.find(marker)
    assert i >= 0, f"見つからない: {marker}"
    return i


def test_who_you_are_is_not_in_the_changing_half():
    """身元はシステム文へ移した（出-e-に）。**プロンプトには残っていない。**

    以前は1本の文字列だったので「先に置く」ことで守っていた。分けたので、順序を人が
    守る必要が無くなった——システム文は常に先である。
    """
    # 中身（差し込み口）はプロンプトから消えている。
    assert "{me}" not in ARBITER_PROMPT
    assert "{family}" not in ARBITER_PROMPT
    # 名前だけは残る。**どこに書いてあるかを指すため**で、そこは「はじめに渡された」と
    # 言い直した（システム文へ移ったので「下の」ではない）。
    assert "はじめに渡された" in ARBITER_PROMPT


def test_the_iteration_cap_note_is_in_the_changing_half():
    """上限の但し書きは5反復に1回だけ現れる（通常は空文字）。

    安定な身元はシステム文にあるので、これが現れてもそちらは作り直しにならない。
    """
    assert "{capped_note}" in ARBITER_PROMPT
    assert _pos("{capped_note}") > _pos("[いま]")


def test_what_the_person_said_comes_last():
    # 毎回変わるものほど後ろ。
    assert _pos("[人の言葉]") > _pos("[いま誰が居るか]")
    assert _pos("[いまの作業状態]") > _pos("[人の言葉]")


def test_the_name_is_not_passed_separately():
    # `ME.md` の「名前： …」が `[あなたは誰か]`（システム文）に入っている。二重に渡さない。
    assert "{agent_name}" not in ARBITER_PROMPT


def test_the_silence_rule_points_at_who_you_are():
    # 名前を別枠で渡さない代わりに、どこに書いてあるかを指す。指す先はシステム文にある。
    assert "あなたは誰か" in ARBITER_PROMPT.split("silence_minutes")[0][-600:]


# ── 立ち位置：調停器はパジュの心そのものである（出-e-に・2026-09-05）──────────

def test_the_arbiter_speaks_as_paju_not_as_a_mechanism():
    """**調停器はパジュの心そのものである。** 自分を機構として名乗らない。

    書き直す前は「あなたは対話エージェントの内部で、次の一手を選ぶ調停器である」と、
    パジュを外から眺めて「その中の部品」として自分を置いていた。`[あなたは誰か]` で
    人格を渡しながら次の行で調停器を名乗るのは食い違いで、PAD 評価が「このやり取りを
    採点せよ」だったのと同じ種類の誤りである。

    **待ってもらう一言と本応答は、同じパジュの2つの出口である。** 実機では、本応答が
    ですますなのに待ってもらう一言だけタメ口になった。同じ人の言葉として揃わなかった。
    """
    from familiar_agent.core.context_parts import Stance, build_context

    assert "対話エージェントの内部で" not in ARBITER_PROMPT
    assert "調停器である" not in ARBITER_PROMPT
    system = build_context(stance=Stance.PAJU, self_understanding="me", family="fam").stable
    assert system.startswith("あなたはパジュである")


def test_the_arbiter_still_answers_only_json():
    """一人称にしても、返すのは JSON だけである（会話ではない）。"""
    assert "JSON だけを返す" in ARBITER_PROMPT
    assert "挨拶や説明はせず" in ARBITER_PROMPT


def test_the_three_branches_are_unchanged():
    """分岐の名前と役割は変えない。書き直すのは自己規定だけである。"""
    for branch in ('"light"', '"full"', '"action"'):
        assert branch in ARBITER_PROMPT, branch
    assert "recall" in ARBITER_PROMPT
    assert "search_deferred" in ARBITER_PROMPT


def test_the_identity_block_matches_what_the_context_mouth_builds():
    """調停の先頭は、文脈の口が組む安定部と**同じ形**である（出-e）。

    構造そのものは寄せられなかった（調停は安定と可変が交互に並び、指示が data の位置に
    合わせて置かれている）。だが**先頭の身元の塊だけは同じ形**にしておく。ここが割れると、
    パジュが場所によって違う名乗り方をすることになる。
    """
    from familiar_agent.core.context_parts import Stance, build_context

    built = build_context(stance=Stance.PAJU, self_understanding="＜私＞", family="＜家＞").stable
    assert built.startswith("あなたはパジュである")
    for block in ("[あなたは誰か]\n＜私＞", "[一緒に暮らす人たち]\n＜家＞"):
        assert block in built, block
    assert built.index("[あなたは誰か]") < built.index("[一緒に暮らす人たち]")


# ── 構造を寄せる：安定はシステム文へ、可変と指示はプロンプトへ（出-e-に）──────

def test_the_stable_identity_goes_into_the_system_text():
    """**1本の文字列だったのは、`complete()` にシステム文の口が無かったからである。**

    出-e-い で口を作ったので分けられる。安定（立ち位置＋人格＋家族）はシステム文へ、
    課題の指示と可変の data はプロンプトへ。**交互に並ぶ問題は消える**——「口調の注意」が
    `[いま]` の直後にあるのも、JSON の指示が最後にあるのも、どちらもプロンプト側の話に
    なるからである。他の6仕事と同じ形になり、システム文は呼び出し間で同一なので
    前方一致キャッシュが最大限効く。
    """
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from familiar_agent.loop.arbiter import arbitrate

    be = MagicMock()
    be.complete = AsyncMock(return_value='{"branch": "full", "effort": "high"}')
    asyncio.run(arbitrate(
        be, utterance="やあ", workspace_ctx="（なし）",
        self_understanding="＜自己認識＞", family_md="＜家族＞",
        present_ctx="（在席）", now_ctx="（いま）", timeout=5.0,
    ))
    system = be.complete.await_args.kwargs["system"]
    prompt = be.complete.await_args.args[0]

    # 安定はシステム文にだけある
    assert system.startswith("あなたはパジュである")
    assert "＜自己認識＞" in system and "＜家族＞" in system
    assert "＜自己認識＞" not in prompt and "＜家族＞" not in prompt

    # 可変と指示はプロンプトにだけある
    for changing in ("（在席）", "（いま）", "やあ"):
        assert changing in prompt, changing
        assert changing not in system, changing
    assert "JSON だけを返す" in prompt
    assert "JSON だけを返す" not in system


def test_the_system_text_repeats_exactly_so_the_cache_can_hit():
    """人の言葉が変わってもシステム文は一字一句同じ。"""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from familiar_agent.loop.arbiter import arbitrate

    seen = []
    for utterance in ("おはよう", "おやすみ"):
        be = MagicMock()
        be.complete = AsyncMock(return_value='{"branch": "full"}')
        asyncio.run(arbitrate(
            be, utterance=utterance, workspace_ctx="（なし）",
            self_understanding="＜自己認識＞", family_md="＜家族＞", timeout=5.0,
        ))
        seen.append(be.complete.await_args.kwargs["system"])
    assert seen[0] == seen[1]
