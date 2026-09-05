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


def test_who_you_are_comes_before_anything_that_changes():
    # `[あなたは誰か]` と `[一緒に暮らす人たち]` は起動中ほぼ変わらない。
    assert _pos("[あなたは誰か]") < _pos("[いま]")
    assert _pos("[一緒に暮らす人たち]") < _pos("[いま]")


def test_the_iteration_cap_note_is_not_in_the_stable_part():
    """上限の但し書きは5反復に1回だけ現れる（通常は空文字）。

    固定のものより前に置くと、現れた回に後ろ全部が作り直しになる。
    """
    assert _pos("{capped_note}") > _pos("[あなたは誰か]")
    assert _pos("{capped_note}") > _pos("[一緒に暮らす人たち]")


def test_what_the_person_said_comes_last():
    # 毎回変わるものほど後ろ。
    assert _pos("[人の言葉]") > _pos("[いま誰が居るか]")
    assert _pos("[いまの作業状態]") > _pos("[人の言葉]")


def test_the_name_is_not_passed_separately():
    # `ME.md` の「名前： …」が `[あなたは誰か]` に入っている。二重に渡さない。
    assert "{agent_name}" not in ARBITER_PROMPT


def test_the_silence_rule_points_at_who_you_are():
    # 名前を別枠で渡さない代わりに、どこに書いてあるかを指す。
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
    assert "対話エージェントの内部で" not in ARBITER_PROMPT
    assert "調停器である" not in ARBITER_PROMPT
    assert "あなたはパジュである" in ARBITER_PROMPT


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

    built = build_context(stance=Stance.PAJU, self_understanding="{me}", family="{family}").stable
    assert ARBITER_PROMPT.startswith("あなたはパジュである")
    for block in ("[あなたは誰か]\n{me}", "[一緒に暮らす人たち]\n{family}"):
        assert block in built, block
        assert block in ARBITER_PROMPT, block
    assert ARBITER_PROMPT.index("[あなたは誰か]") < ARBITER_PROMPT.index("[一緒に暮らす人たち]")
