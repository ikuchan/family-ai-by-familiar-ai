"""文脈を組む口（出-e）。どの仕事へどの部品を、どの順で渡すか。

同じ組み立てが `build_event_system_prompt` と `ARBITER_PROMPT` の2箇所にあり、**同じ
不変条件を別々に手で守っていた**。「起動中ほぼ変わらないもの（人格・家族・規則）を先に
置き、変わるもの（時刻・在席・人の言葉・作業状態）を後ろへ」——前方一致キャッシュが効く
条件である。守れなかったとき実機で調停が時間切れになり、沈黙依頼が読まれないまま倒れた。
3箇所目を手で書けば同じ芽をもう一つ作るので、口を1つにする。

**感情を作るのはパジュである。** 立ち位置は仕事で分かれ、一人称は単独では成り立たない
（自分が誰で誰と暮らしているかを知らなければパジュにはなれない）。外から測る立ち位置は
逆に人格を伴わない——整合チェックをパジュとしてやらせると、違反18件中0〜1件しか
捕まえなかった（`根拠台帳` §25.8）。
"""

from __future__ import annotations

import pytest

from familiar_agent.core.context_parts import Stance, build_context
from familiar_agent.loop.prompt import EVENT_SYSTEM_PROMPT, rules_section

_ME = "# 私について\n名前： パジュ\n性格：好奇心旺盛"
_FAMILY = "# 家族\nゆうすけ（大人）\nはるか（子ども）"


# ── 規則の節を切り出す ──────────────────────────────────────────────────────

def test_the_rules_section_is_cut_out_by_matching_parentheses():
    """行数や位置ではなく括弧の対応で切る。S 式が編集されても壊れない。"""
    sec = rules_section()
    assert sec.startswith("  (rules")
    assert sec.rstrip().endswith('"相手が使った言語で応答する。"))')
    assert sec in EVENT_SYSTEM_PROMPT
    assert sec.count("(") == sec.count(")")


# ── 一人称は単独では成り立たない ────────────────────────────────────────────

def test_speaking_as_paju_needs_the_self_and_the_family():
    ctx = build_context(stance=Stance.PAJU, self_understanding=_ME, family=_FAMILY)
    assert "パジュ" in ctx.stable
    assert _ME in ctx.stable
    assert _FAMILY in ctx.stable


@pytest.mark.parametrize("missing", ["self_understanding", "family"])
def test_speaking_as_paju_without_a_part_cannot_be_built(missing):
    kw = {"self_understanding": _ME, "family": _FAMILY}
    kw[missing] = ""
    with pytest.raises(ValueError, match="パジュ"):
        build_context(stance=Stance.PAJU, **kw)


def test_measuring_from_outside_needs_neither():
    ctx = build_context(stance=Stance.INSTRUMENT)
    assert "パジュ自身ではない" in ctx.stable
    assert _ME not in ctx.stable


# ── 選んだ部品だけが入る ────────────────────────────────────────────────────

def test_only_the_chosen_parts_are_present():
    ctx = build_context(stance=Stance.INSTRUMENT, rules="（規則）")
    assert "（規則）" in ctx.stable
    assert _FAMILY not in ctx.stable
    assert ctx.variable == ""


# ── 安定が先、可変が後 ──────────────────────────────────────────────────────

def test_the_changing_parts_stay_out_of_the_stable_half():
    ctx = build_context(
        stance=Stance.PAJU, self_understanding=_ME, family=_FAMILY,
        now="(now :datetime \"2026-09-05 12:00\")", presence="（在席）", workspace="（作業状態）",
    )
    for changing in ("2026-09-05", "（在席）", "（作業状態）"):
        assert changing not in ctx.stable, changing
        assert changing in ctx.variable, changing


def test_the_variable_half_keeps_the_documented_order():
    """いま → 在席 → 反復 → 作業状態。既存2箇所が守っていた並びである。"""
    ctx = build_context(
        stance=Stance.INSTRUMENT, now="ＮＯＷ", presence="ＩＮ",
        iteration="ＩＴ", workspace="ＷＳ",
    )
    v = ctx.variable
    assert v.index("ＮＯＷ") < v.index("ＩＮ") < v.index("ＩＴ") < v.index("ＷＳ")


def test_the_stable_half_repeats_exactly_so_the_cache_can_hit():
    """同じ部品なら安定部は一字一句同じ。可変部が変わっても影響しない。"""
    a = build_context(stance=Stance.PAJU, self_understanding=_ME, family=_FAMILY, now="朝")
    b = build_context(stance=Stance.PAJU, self_understanding=_ME, family=_FAMILY, now="夜")
    assert a.stable == b.stable
    assert a.variable != b.variable


def test_a_context_with_nothing_variable_has_an_empty_variable_half():
    assert build_context(stance=Stance.INSTRUMENT).variable == ""


# ── 静的核を渡すなら、立ち位置の一文は置かない ──────────────────────────────

def test_the_static_core_replaces_the_stance_line():
    """静的核の `(identity ...)` が同じことを厚く言っている。二度書かない。"""
    ctx = build_context(
        stance=Stance.PAJU, core="（静的核）",
        self_understanding=_ME, family=_FAMILY,
    )
    assert ctx.stable.startswith("[身体と決まり]")
    assert "あなたはパジュである" not in ctx.stable
    assert "（静的核）" in ctx.stable


def test_without_the_static_core_the_stance_line_leads():
    """軽量LLM には静的核を渡さないので、立ち位置の一文が代わりを務める。"""
    ctx = build_context(stance=Stance.PAJU, self_understanding=_ME, family=_FAMILY)
    assert ctx.stable.startswith("あなたはパジュである")


def test_the_stable_half_is_labelled():
    ctx = build_context(
        stance=Stance.INSTRUMENT, core="（静的核）", self_understanding=_ME,
        family=_FAMILY, rules="（規則）",
    )
    for label in ("[身体と決まり]", "[あなたは誰か]", "[一緒に暮らす人たち]", "[守っている決まり]"):
        assert label in ctx.stable, label
    i = [ctx.stable.index(x) for x in
         ("[身体と決まり]", "[あなたは誰か]", "[一緒に暮らす人たち]", "[守っている決まり]")]
    assert i == sorted(i)


def test_the_inner_state_sits_between_presence_and_iteration():
    """主LLM の可変部の並び：いま → 在席 → 内部状態 → 反復 → 作業状態。"""
    ctx = build_context(
        stance=Stance.INSTRUMENT, now="ＮＯＷ", presence="ＩＮ",
        inner_state="ＰＩ", iteration="ＩＴ", workspace="ＷＳ",
    )
    v = ctx.variable
    assert v.index("ＮＯＷ") < v.index("ＩＮ") < v.index("ＰＩ") < v.index("ＩＴ") < v.index("ＷＳ")


# ── 静的核があるなら、欠けていても組める ────────────────────────────────────

def test_the_static_core_carries_the_identity_so_parts_may_be_missing():
    """身元を担うのが静的核なら、自己認識や家族が無くても組める。

    `FAMILY.md` が無い機体や、自己認識をまだ生成していない初回起動がある。そこで
    落とすと**主LLM のターンごと落ちる**。要求が要るのは、立ち位置の一文しか身元が
    無いとき（＝軽量LLM）である。
    """
    ctx = build_context(stance=Stance.PAJU, core="（静的核）")
    assert ctx.stable.startswith("[身体と決まり]")
    ctx2 = build_context(stance=Stance.PAJU, core="（静的核）", self_understanding=_ME)
    assert _ME in ctx2.stable
