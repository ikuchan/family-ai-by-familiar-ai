"""調停の文の並び：固定のものを先、変わるものを後へ。

プロンプトキャッシュは**前方一致**で効く。変わりうるものが固定のものより前にあると、それが
変わるたびに後ろ全部が作り直しになる。

実機で、調停が 2 秒で返らず時間切れになった（沈黙依頼が読まれないまま `full` へ倒れた）。
渡すものを増やしてきたので、キャッシュが効く形に組み直した。

出-au 段 5-7d で、調停は Jev の判定（`Arbiter._state`）と軽量LLM の文章（`WRITER_PROMPT`）に分かれた。
人格と家族を持つのは文章の側だけで、システム文で渡す。名前は `[あなたは誰か]`（`ME.md`）に既に入っている。
同じものを別枠でもう一度渡さない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.arbiter import (
    _CAPPED_NOTE,
    WRITER_PROMPT,
    Arbiter,
    ArbiterInput,
)


def _pos(marker: str) -> int:
    i = WRITER_PROMPT.find(marker)
    assert i >= 0, f"見つからない: {marker}"
    return i


def _written(utterance="やあ"):
    """light の返事を書かせ、軽量LLM へ渡ったシステム文とプロンプトを返す。"""
    w = MagicMock()
    w.complete = AsyncMock(return_value='{"text": "うん"}')
    inp = ArbiterInput(
        utterance=utterance,
        workspace_ctx="（なし）",
        self_understanding="＜自己認識＞",
        family_md="＜家族＞",
        present_ctx="（在席）",
        now_ctx="（いま）",
    )
    asyncio.run(Arbiter(jev=None, writer=w, timeout=5.0)._write(inp, {"branch": "light"}))
    return w.complete.await_args.kwargs["system"], w.complete.await_args.args[0]


def test_who_you_are_is_not_in_the_changing_half():
    """身元はシステム文にある（出-e-に）。**プロンプトには差し込み口が無い。**"""
    assert "{me}" not in WRITER_PROMPT
    assert "{family}" not in WRITER_PROMPT
    # 名前は、どこに書いてあるかを指すだけ（システム文なので「はじめに渡された」）。
    assert "はじめに渡された" in WRITER_PROMPT


def test_the_iteration_cap_note_goes_to_the_jev_state():
    """上限の但し書きは Jev に送る文に載る（通常は載らない）。分岐を決めるのは Jev である。"""
    a = Arbiter(jev=None, writer=None)
    note = _CAPPED_NOTE.strip()
    assert note in a._state(ArbiterInput(utterance="x", workspace_ctx="", capped=True))
    assert note not in a._state(ArbiterInput(utterance="x", workspace_ctx=""))


def test_what_the_person_said_comes_after_who_is_here():
    # 毎回変わるものほど後ろ。見出しは起点で差し替わる（`[人の言葉]`／`[いま湧いたこと]`・情-e）
    # ので、雛形では `{heading}` の位置で見る。
    assert _pos("{heading}") > _pos("[いま誰が居るか]")
    assert _pos("[いまの作業状態]") > _pos("{heading}")


def test_the_name_is_not_passed_separately():
    # `ME.md` の「名前： …」が `[あなたは誰か]`（システム文）に入っている。二重に渡さない。
    assert "{agent_name}" not in WRITER_PROMPT


# ── 立ち位置：調停はパジュの心そのものである（出-e-に・2026-09-05）──────────


def test_the_arbiter_speaks_as_paju_not_as_a_mechanism():
    """**調停はパジュの心そのものである。** 自分を機構として名乗らない。

    **待ってもらう一言と本応答は、同じパジュの2つの出口である。** 実機では、本応答が
    ですますなのに待ってもらう一言だけタメ口になった。同じ人の言葉として揃わなかった。
    """
    assert "対話エージェントの内部で" not in WRITER_PROMPT
    assert "調停器である" not in WRITER_PROMPT
    system, _ = _written()
    assert system.startswith("あなたはパジュである")


def test_the_writer_still_answers_only_json():
    """一人称にしても、返すのは JSON だけである（会話ではない）。"""
    assert "JSON だけを返す" in WRITER_PROMPT
    assert "挨拶や説明はせず" in WRITER_PROMPT


def test_the_three_branches_are_unchanged():
    """分岐の名前と役割は変えない（Jev の選択肢）。"""
    qs = Arbiter(jev=None, writer=None)._questions(ArbiterInput(utterance="x", workspace_ctx=""))
    assert set(qs["branch"]["criteria"]) >= {"light", "full", "action"}


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
    """安定（立ち位置＋人格＋家族）はシステム文へ、課題の指示と可変の data はプロンプトへ。

    システム文は呼び出し間で同一なので前方一致キャッシュが最大限効く。
    """
    system, prompt = _written()

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
    assert _written("おはよう")[0] == _written("おやすみ")[0]
