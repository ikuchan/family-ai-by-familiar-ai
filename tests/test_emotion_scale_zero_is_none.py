"""快と不快の目盛りを「0＝無い」に揃える（案A）。

以前の感情の評価の指示文は P と Pn について「0 none」と「0.5 neutral」を同居させており、
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
from familiar_agent.core import jev_judges
from familiar_agent.mood_register import MoodPAD, decay_to_rest

_NEUTRAL_P = 0.10
_NEUTRAL_PN = 0.10


# ── 段1：口の目盛り ─────────────────────────────────────────────────────────


# 出-au 段 5-6 で感情の評価は Jev の Score（5 段）へ移した。目盛りの決まりは段の並びと送る文（state）で守る。


def test_the_pleasure_axes_start_at_none():
    """快と不快の段は「0＝無い」から始まる。真ん中を「中立」と書かない。"""
    assert jev_judges._AMOUNT_LEVELS[0] == "まったく無い"
    assert "中立" not in "".join(jev_judges._AMOUNT_LEVELS)


def test_the_dominance_axis_keeps_its_midpoint():
    assert jev_judges._DOM_LEVELS[2] == "ふつう"


def test_the_state_says_where_the_axes_rest_and_asks_what_paju_felt():
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    c = MagicMock(available=True)
    c.ask = AsyncMock(return_value=None)
    asyncio.run(
        jev_judges.judge_emotion(c, text="やった！", mood=MoodPAD(), arousal=0.9, a_gate=0.25)
    )
    state, questions = c.ask.await_args.args
    assert "快 0.10・不快 0.10・掌握 0.50" in state
    assert all("パジュ自身が感じた" in q["instructions"] for q in questions.values())


# ── 段2：中立と気分の減衰先 ───────────────────────────────────────────────────


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
        _NEUTRAL_P,
        _NEUTRAL_PN,
        0.50,
        0.50,
    )


def test_mood_decays_to_the_per_axis_rest_point():
    """半減期ちょうどで、各軸が自分の戻り先との距離を半分にする。"""
    m = MoodPAD(p=0.90, pn=0.90, a=0.90, dom=0.90)
    out = decay_to_rest(m, 600.0)
    assert out.p == 0.5 * (0.90 + _NEUTRAL_P)  # 0.50
    assert out.pn == 0.5 * (0.90 + _NEUTRAL_PN)  # 0.50
    assert out.a == 0.5 * (0.90 + 0.50)  # 0.70
    assert out.dom == 0.5 * (0.90 + 0.50)  # 0.70


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
