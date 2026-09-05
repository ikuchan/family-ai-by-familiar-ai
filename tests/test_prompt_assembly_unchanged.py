"""主LLM のシステムプロンプトの組み立て（出-e で口へ寄せたあと）。

**書式は意図して変えた**（2026-09-05・案ロ-1）。区切りの `---` をやめ、安定部の3つへ
見出しを付ける。既存2箇所が別々の書式で同じ不変条件を守っていたので、口へ寄せるには
どちらかへ揃える必要があった。**立ち位置の一文は置かない**——静的核の `(identity ...)` が
同じことを厚く言っており、二度書くことになる。

**並びは変えていない。** 事故の原因になったのは書式ではなく並びである。

`ARBITER_PROMPT` の並びは実機の事故から来ている。調停が2秒で返らず時間切れになり、
沈黙依頼が読まれないまま倒れた。前方一致キャッシュが効くよう、起動中ほぼ変わらないものを
先に置く。**その並びが崩れていないことを、名前の出現順で確かめる。**
"""

from __future__ import annotations

from familiar_agent.loop.arbiter import ARBITER_PROMPT
from familiar_agent.loop.prompt import build_event_system_prompt

_ME = "私は パジュ である。"
_FAMILY = "ゆうすけ（大人）"


def test_the_event_system_prompt_is_labelled_and_ordered():
    stable, variable = build_event_system_prompt(
        self_understanding=_ME,
        family_md=_FAMILY,
        present_ctx="（在席）",
        pi_ctx="（内部状態）",
        workspace_ctx="（作業状態）",
        iter_ctx="[反復] 1/3",
    )
    # 安定部＝静的核 → 自己認識 → FAMILY。見出し付き、立ち位置の一文は無し。
    assert stable.startswith("[身体と決まり]")
    assert "あなたはパジュである" not in stable
    assert "(agent :type embodied" in stable
    assert stable.index("[身体と決まり]") < stable.index("[あなたは誰か]") < stable.index(
        "[一緒に暮らす人たち]")
    assert stable.index(_ME) < stable.index(_FAMILY)
    # 可変部は日時 → 在席 → 内部状態 → 反復 → 作業状態
    for a, b in zip(("(now :datetime", "（在席）", "（内部状態）", "[反復] 1/3"),
                    ("（在席）", "（内部状態）", "[反復] 1/3", "（作業状態）")):
        assert variable.index(a) < variable.index(b), (a, b)
    # 変わるものが安定部へ混じらない（キャッシュが効かなくなる）
    for changing in ("（在席）", "（作業状態）", "[反復] 1/3"):
        assert changing not in stable, changing


def test_the_arbiter_no_longer_needs_to_order_the_stable_part_by_hand():
    """安定部はシステム文へ移した（出-e-に）。**順序を人が守る必要が無くなった。**

    以前は1本の文字列だったので「人格・家族を先に置く」ことでキャッシュを効かせていた。
    分けたので、システム文は常に先である。プロンプトに残るのは可変と課題の指示だけで、
    その中の並び（時刻 → 在席 → 言葉 → 作業状態）は変えていない。
    """
    assert "{me}" not in ARBITER_PROMPT
    assert "{family}" not in ARBITER_PROMPT
    i_now = ARBITER_PROMPT.index("{now}")
    i_present = ARBITER_PROMPT.index("{present}")
    i_utterance = ARBITER_PROMPT.index("{utterance}")
    i_workspace = ARBITER_PROMPT.index("{workspace}")
    assert i_now < i_present < i_utterance < i_workspace


def test_the_arbiter_prompt_places_the_rules_before_the_moment():
    """判断の規則（分岐の説明）は、いつ・誰・何を言われたかより前にある。"""
    assert ARBITER_PROMPT.index('"light"') < ARBITER_PROMPT.index("{now}")
