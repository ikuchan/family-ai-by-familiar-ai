"""測っていない感情を器が持たないこと（②）と、気分変調の原点（①）。

案A で快と不快の平静を 0.10 へ移したことの後始末である。二つとも、0.5 を絶対の原点と
みなしていた箇所が表に出たものである。

②`_row_to_mental_item` は `row.get("emotion_p", 0.5)` で列を読んでいた。`dict.get` は
**キーがあって値が None なら None を返す**ので、050 が NULL を入れられるようにした時点から
「列を SELECT していない（分からない）」と「列が NULL（測っていない）」を取り違えていた。
`用語_略語一覧` の PI 項は「評価前は未設定で持つ（評価結果としての中立と、未評価の未設定とを
区別するため）」と定めており、0.5 で埋めるのはこれに反する。A（高ぶり）は機械値で常に実値が
あるので捨てず、別の欄で持つ。

①`_gain` は `logit(x)` を絶対値で足していた。これが消えるのは `x = 0.5` のときだけなので、
平静が 0.10 へ動くと「中立気分では g = b_i」（`感情ループ全体像`）が成り立たなくなる。
各項を平静からのずれにして、設計の記述をそのまま保つ。
"""

from __future__ import annotations

import pytest

from familiar_agent.config import DriveConfig
from familiar_agent.core.drive_dynamics import g_d
from familiar_agent.core.mental_item import MentalItem, _row_to_mental_item
from familiar_agent.mood_register import REST_PAD, MoodPAD
from familiar_agent.tif import build_primitive, expand_to_mental

_CFG = DriveConfig()


def _row(**over):
    row = {"id": "x", "content": "c", "superseded_by": None, "groundedness_g0": 1.0}
    row.update(over)
    return row


# ── ② 器 ────────────────────────────────────────────────────────────────────

def test_a_fully_measured_row_carries_both_the_pad_and_the_arousal():
    item = _row_to_mental_item(_row(
        emotion_p=0.8, emotion_pn=0.15, emotion_a=0.72, emotion_dom=0.6))
    assert item.emotion == MoodPAD(0.8, 0.15, 0.72, 0.6)
    assert item.arousal == 0.72


def test_an_unmeasured_row_keeps_the_arousal_and_drops_the_feeling():
    """P/Pn/Dom が NULL でも A は機械値として残る（050）。感情は埋めない。"""
    item = _row_to_mental_item(_row(
        emotion_p=None, emotion_pn=None, emotion_a=0.5, emotion_dom=None))
    assert item.emotion is None
    assert item.arousal == 0.5


def test_one_missing_axis_is_enough_to_call_the_feeling_unmeasured():
    item = _row_to_mental_item(_row(
        emotion_p=0.8, emotion_pn=0.15, emotion_a=0.72, emotion_dom=None))
    assert item.emotion is None
    assert item.arousal == 0.72


def test_a_row_read_without_the_columns_has_neither():
    """列を SELECT していなければ、感情も高ぶりも分からない。中立で埋めない。"""
    item = _row_to_mental_item(_row())
    assert item.emotion is None
    assert item.arousal is None


# 守り：PI は変えない（A は I が取り込みで作る機械値で、T の「感じ＋欲」ではない）
def test_the_primitive_item_is_unchanged():
    from familiar_agent.drive_register import AiDrivers

    pi = build_primitive(MoodPAD(0.9, 0.1, 0.8, 0.6), AiDrivers())
    assert pi.emotion == MoodPAD(0.9, 0.1, 0.8, 0.6)
    assert not hasattr(pi, "arousal")
    mi = expand_to_mental(pi, id="x", content="y")
    assert isinstance(mi, MentalItem)
    assert mi.emotion == MoodPAD(0.9, 0.1, 0.8, 0.6)


# ── ① 気分変調の原点 ────────────────────────────────────────────────────────

def test_at_rest_the_modulation_vanishes():
    """平静では g = b_i（`感情ループ全体像`）。平静が動いてもこの性質は保つ。"""
    g = g_d(REST_PAD)
    assert g.seeking == pytest.approx(_CFG.bias_seeking, abs=1e-9)
    assert g.rest == pytest.approx(_CFG.bias_rest, abs=1e-9)
    assert g.bond == pytest.approx(_CFG.bias_bond, abs=1e-9)
    assert g.safety == pytest.approx(_CFG.bias_safety, abs=1e-9)
    assert g.esteem == pytest.approx(_CFG.bias_esteem, abs=1e-9)


# 守り：式の一般化であって作り替えではない
def test_the_old_origin_reproduces_the_old_behaviour():
    """平静を旧 0.5 に指定すれば、旧実装と同じ値が出る。"""
    old_rest = MoodPAD(0.5, 0.5, 0.5, 0.5)
    g = g_d(old_rest, rest=old_rest)
    assert g.bond == pytest.approx(_CFG.bias_bond, abs=1e-9)


def test_the_direction_of_the_coefficients_is_unchanged():
    """BOND は P が低いほど募る（寂しい）。係数の意味は変えていない。"""
    lonely = g_d(MoodPAD(p=0.05, pn=REST_PAD.pn, a=REST_PAD.a, dom=REST_PAD.dom))
    content = g_d(MoodPAD(p=0.90, pn=REST_PAD.pn, a=REST_PAD.a, dom=REST_PAD.dom))
    assert lonely.bond > content.bond
