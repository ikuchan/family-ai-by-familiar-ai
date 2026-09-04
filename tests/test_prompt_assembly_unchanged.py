"""既存の2箇所を出-e の口へ寄せても、出る文字列が変わらないこと。

**動いているものを触るので、先に現状を固定する。** ここが落ちたら、寄せ方が挙動を
変えている。環-e が「挙動を変えない」で進めているのと同じやり方である。

`ARBITER_PROMPT` の並びは実機の事故から来ている。調停が2秒で返らず時間切れになり、
沈黙依頼が読まれないまま倒れた。前方一致キャッシュが効くよう、起動中ほぼ変わらないものを
先に置く。**その並びが崩れていないことを、名前の出現順で確かめる。**
"""

from __future__ import annotations

from familiar_agent.loop.arbiter import ARBITER_PROMPT
from familiar_agent.loop.prompt import build_event_system_prompt

_ME = "私は パジュ である。"
_FAMILY = "ゆうすけ（大人）"


def test_the_event_system_prompt_is_assembled_as_before():
    stable, variable = build_event_system_prompt(
        self_understanding=_ME,
        family_md=_FAMILY,
        present_ctx="（在席）",
        pi_ctx="（内部状態）",
        workspace_ctx="（作業状態）",
        iter_ctx="[反復] 1/3",
    )
    # 安定部＝静的核 → 自己認識 → FAMILY を区切りでつないだもの
    assert stable.startswith("(agent :type embodied")
    assert stable.index(_ME) < stable.index(_FAMILY)
    assert "\n\n---\n\n" in stable
    # 可変部は日時 → 在席 → 内部状態 → 反復 → 作業状態
    for a, b in zip(("(now :datetime", "（在席）", "（内部状態）", "[反復] 1/3"),
                    ("（在席）", "（内部状態）", "[反復] 1/3", "（作業状態）")):
        assert variable.index(a) < variable.index(b), (a, b)
    # 変わるものが安定部へ混じらない（キャッシュが効かなくなる）
    for changing in ("（在席）", "（作業状態）", "[反復] 1/3"):
        assert changing not in stable, changing


def test_the_arbiter_prompt_keeps_stable_before_changing():
    """人格・家族・規則が先、時刻・在席・言葉・作業状態が後（前方一致キャッシュ）。"""
    order = ["[あなたは誰か]", "[一緒に暮らす人たち]", "{me}", "{family}"]
    for name in order:
        assert name in ARBITER_PROMPT, name
    i_me = ARBITER_PROMPT.index("{me}")
    i_family = ARBITER_PROMPT.index("{family}")
    i_now = ARBITER_PROMPT.index("{now}")
    i_utterance = ARBITER_PROMPT.index("{utterance}")
    i_workspace = ARBITER_PROMPT.index("{workspace}")
    assert i_me < i_family < i_now
    assert i_now < i_utterance < i_workspace


def test_the_arbiter_prompt_places_the_rules_before_the_moment():
    """判断の規則（分岐の説明）は、いつ・誰・何を言われたかより前にある。"""
    assert ARBITER_PROMPT.index('"light"') < ARBITER_PROMPT.index("{now}")
