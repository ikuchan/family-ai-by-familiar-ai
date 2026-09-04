"""快と不快の目盛りを「0＝無い」に揃える（案A）。

`_EMOTION_PAD_PROMPT` は P と Pn について「0 none」と「0.5 neutral」を同居させており、
中立の出来事をどちらに置くかが決まっていなかった。実測でモデル間が 0.37 開いた
（事務連絡の Pn が gemini-2.5-flash-lite 0.08 対 claude-haiku-4-5 0.45・根拠台帳 §25.4）。

`LABEL_PAD` の12点は `neutral` を除いてすでに「0＝無い」で書かれている。`nostalgic`
(0.55, 0.55) と `moved` (0.75, 0.50) は「両方の軸が中程度ある＝ほろ苦い」であって、
「0.5＝中立」では読めない。だからこの変更は新しい目盛りの持ち込みではなく、
**プロンプトと `neutral` と mood の減衰先を、残り11点へ揃える**ことである。

中立は P=0.10 / Pn=0.10、A と Dom は 0.50 のまま（Dom は 0＝無力 ↔ 1＝掌握で両極が
揃っており、目盛りの矛盾が無い）。
"""

from __future__ import annotations

from familiar_agent.emotion_pad import LABEL_PAD, label_from_pad
from familiar_agent.loop.evaluator import _EMOTION_PAD_PROMPT
from familiar_agent.mood_register import MoodPAD, decay_to_rest

_NEUTRAL_P = 0.10
_NEUTRAL_PN = 0.10


# ── 段1：口の目盛り ─────────────────────────────────────────────────────────

def _axis_line(head: str) -> str:
    return next(ln for ln in _EMOTION_PAD_PROMPT.splitlines() if ln.lstrip().startswith(head))


def test_the_pleasure_axes_have_no_midpoint_anchor():
    """快と不快は片側の量。0 が「無い」で、真ん中に印を置かない。

    どちらも 0＝まったく無い ↔ 1＝とても大きい で、0.5 に名前が付いていないこと。
    印を置くと、中立の出来事をどこへ置くかが決まらなくなる（それが案A の出発点だった）。
    """
    for head in ("- P ", "- Pn"):
        line = _axis_line(head)
        assert "まったく無い" in line, line
        assert "0.5" not in line, line


def test_the_dominance_axis_keeps_its_midpoint():
    """Dom は 0＝無力 ↔ 1＝掌握 の両極なので 0.5 が中点で正しい。ここは変えない。"""
    line = _axis_line("- Dom")
    assert "0.5" in line, line


def test_the_prompt_states_where_the_axes_rest():
    """平静の位置は事実として伝える（指示ではない）。無いと中立が真ん中へ寄る。"""
    assert "P=0.10 / Pn=0.10 / Dom=0.50" in _EMOTION_PAD_PROMPT


def test_the_prompt_asks_what_paju_felt_rather_than_a_rating():
    """感情を作るのはパジュである。外から採点させない。"""
    assert "あなた自身が何を感じたか" in _EMOTION_PAD_PROMPT
    assert "Rate the emotion" not in _EMOTION_PAD_PROMPT


def test_the_prompt_says_nobody_else_reads_it():
    assert "あなた以外だれも見ない" in _EMOTION_PAD_PROMPT


# ── 段2：中立の位置 ─────────────────────────────────────────────────────────

def test_the_neutral_label_sits_where_neither_feeling_is_present():
    assert LABEL_PAD["neutral"] == (_NEUTRAL_P, _NEUTRAL_PN, 0.50, 0.50)


def test_a_default_mood_is_neither_pleasant_nor_unpleasant():
    m = MoodPAD()
    assert (m.p, m.pn) == (_NEUTRAL_P, _NEUTRAL_PN)
    assert (m.a, m.dom) == (0.50, 0.50)


def test_the_rest_point_is_per_axis():
    # 軸ごとの戻り先。単一の `REST` を置き換える（収集を止めないよう関数内で引く）。
    from familiar_agent.mood_register import REST_PAD

    assert (REST_PAD.p, REST_PAD.pn, REST_PAD.a, REST_PAD.dom) == (
        _NEUTRAL_P, _NEUTRAL_PN, 0.50, 0.50
    )


def test_mood_decays_to_the_per_axis_rest_point():
    """半減期ちょうどで、各軸が自分の戻り先との距離を半分にする。"""
    m = MoodPAD(p=0.90, pn=0.90, a=0.90, dom=0.90)
    out = decay_to_rest(m, 600.0)
    assert out.p == 0.5 * (0.90 + _NEUTRAL_P)      # 0.50
    assert out.pn == 0.5 * (0.90 + _NEUTRAL_PN)    # 0.50
    assert out.a == 0.5 * (0.90 + 0.50)            # 0.70
    assert out.dom == 0.5 * (0.90 + 0.50)          # 0.70


def test_an_unmeasured_mood_json_falls_back_per_axis():
    """欄が欠けた保存値は、軸ごとの戻り先で埋める（単一の 0.5 ではない）。"""
    m = MoodPAD.from_json_dict({})
    assert (m.p, m.pn, m.a, m.dom) == (_NEUTRAL_P, _NEUTRAL_PN, 0.50, 0.50)


def test_the_neutral_point_still_reads_as_neutral():
    assert label_from_pad(MoodPAD()) == "neutral"


# ── 守り：壊していないことの確認（実装の前後どちらでも通るべき）──────────────

def test_the_other_eleven_labels_are_untouched():
    """`neutral` 以外は動かさない。11点はもともと「0＝無い」で書かれている。"""
    assert LABEL_PAD["happy"] == (0.80, 0.15, 0.55, 0.60)
    assert LABEL_PAD["sad"] == (0.20, 0.75, 0.25, 0.30)
    assert LABEL_PAD["nostalgic"] == (0.55, 0.55, 0.30, 0.45)
    assert len(LABEL_PAD) == 12


def test_the_exact_label_points_still_map_to_themselves():
    assert label_from_pad(MoodPAD(0.80, 0.15, 0.55, 0.60)) == "happy"
    assert label_from_pad(MoodPAD(0.20, 0.75, 0.25, 0.30)) == "sad"
    assert label_from_pad(MoodPAD(0.75, 0.15, 0.55, 0.90)) == "proud"
