"""記憶接続 OIF が通す型（`設計方針_OIF` v0.1）。

`observations` の列を読み手ごとに調べて MI を 18 属性で決めた。主想起・拡散想起・
プロンプトのいずれからも読まれない列は入れず、計算で作れるものも入れない。

- `kind` は `direction` から決まる（12 種の `direction` に対し `kind` は 6 種で、
  8 つの `direction` がすべて `observation` に落ちる）。
- `emotion_vec` は PAD から作る索引で、意味としては PAD と同じもの。
- 採点（`fit` ほか）は想起のたびに算出する導出値で、保存しない。
- `person_id`・`scope`・`importance` は読み手が居ないか旧い（撤去は 記-d）。

`View`（見方）は「どう探すか」で、書き込み側の**視点**（`writer_id`・`subject_id`・
`participants`）とは別のものである。語を分けないと中身が混ざる。
"""

from __future__ import annotations

import dataclasses
from datetime import datetime

import pytest

from familiar_agent.io.oif import MI, Cue, Health, Recalled, Span, Verdict, View
from familiar_agent.mood_register import MoodPAD

_EXPECTED = {
    "id", "content", "timestamp",
    "direction", "emotion",
    "parent_id", "superseded_by",
    "pad",
    "groundedness_g0", "groundedness_n",
    "last_recalled_at",
    # 面の同定（案3）。視点3属性（`writer_id`／`subject_id`／`participants`）が
    # ここへ置き換わった。誰との関係かは面が持つ。
    "obs_id", "person_id", "relation_key",
    "image_path", "image_data",
}


class TestMI:
    def test_the_attributes_are_fixed(self) -> None:
        got = {f.name for f in dataclasses.fields(MI)}
        assert got == _EXPECTED, (
            f"欠け: {_EXPECTED - got}\n余り: {got - _EXPECTED}"
        )

    def test_derived_values_are_not_attributes(self) -> None:
        """計算で作れるものは持たない。"""
        got = {f.name for f in dataclasses.fields(MI)}
        for name in ("kind", "emotion_vec", "vector", "fit", "score", "groundedness"):
            assert name not in got, f"{name} は導出値なので属性に持たない"

    def test_removed_columns_are_not_attributes(self) -> None:
        """撤去した列は持たない。

        `scope` と `importance` は 記-d が 039 で、`recall_count` は 043 が、
        所有者の `person_id` は 042 が撤去した。
        """
        got = {f.name for f in dataclasses.fields(MI)}
        for name in ("scope", "importance", "recall_count"):
            assert name not in got, f"{name} は撤去した列"

    def test_the_perspective_columns_became_the_facet(self) -> None:
        """視点3属性は面へ置き換わった（案3）。

        `MI.person_id` は**所有者ではなく、誰との関係か**である。042 が撤去したのは
        `observations.person_id`（所有者）で、面の `person_id` とは別のものである。
        """
        got = {f.name for f in dataclasses.fields(MI)}
        for name in ("writer_id", "subject_id", "participants"):
            assert name not in got, f"{name} は面が引き取った"
        assert {"obs_id", "person_id", "relation_key"} <= got

    def test_kind_is_derived_from_direction(self) -> None:
        """`kind` は `direction` から決まる。"""
        mi = MI(id="x", content="c", timestamp=datetime.now(), direction="会話")
        assert mi.kind == "conversation"
        assert MI(id="y", content="c", timestamp=datetime.now(),
                  direction="内省").kind == "self_model"
        assert MI(id="z", content="c", timestamp=datetime.now(),
                  direction="好奇心").kind == "curiosity"

    def test_unknown_direction_falls_back_to_observation(self) -> None:
        """表に無い direction は observation に落とす（8つの direction がそうなっている）。"""
        mi = MI(id="x", content="c", timestamp=datetime.now(), direction="求め")
        assert mi.kind == "observation"

    def test_the_two_version_columns_are_separate(self) -> None:
        """`parent_id`（過去へ）と `superseded_by`（未来へ）は別のもの。"""
        mi = MI(id="x", content="c", timestamp=datetime.now(), direction="発話",
                parent_id="起点", superseded_by="次の版")
        assert mi.parent_id == "起点"
        assert mi.superseded_by == "次の版"

    def test_pad_defaults_to_neutral(self) -> None:
        mi = MI(id="x", content="c", timestamp=datetime.now(), direction="観察")
        assert isinstance(mi.pad, MoodPAD)
        assert mi.pad.p == pytest.approx(0.5)


class TestCue:
    def test_a_bare_cue_is_just_text(self) -> None:
        assert Cue(text="昨日の天気").text == "昨日の天気"

    def test_the_ways_of_looking_are_separate_fields(self) -> None:
        """日付・月日・種別は別の欄（いま別メソッドになっているものをまとめる）。"""
        got = {f.name for f in dataclasses.fields(Cue)}
        assert got == {"text", "direction", "on_date", "on_month_day",
                       "exclude", "open_ids"}, got

    def test_a_cue_is_frozen(self) -> None:
        cue = Cue(text="x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            cue.text = "y"          # type: ignore[misc]


class TestView:
    def test_the_defaults_come_from_config(self) -> None:
        """既定は課題5 の確定値（K=7・床 0.05）。"""
        v = View()
        assert v.k == 7
        assert v.floor == pytest.approx(0.05)

    def test_a_view_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            View().k = 3            # type: ignore[misc]


class TestRecalled:
    def test_it_carries_the_score_beside_the_mi(self) -> None:
        """採点は MI の中でなく、想起結果が添える。"""
        mi = MI(id="x", content="c", timestamp=datetime.now(), direction="会話")
        r = Recalled(mi=mi, fit=0.42, groundedness=0.7)
        assert r.mi is mi
        assert r.fit == pytest.approx(0.42)
        assert not hasattr(r.mi, "fit"), "採点が MI へ混ざっている"


class TestVerdict:
    def test_the_four_values(self) -> None:
        assert {v.value for v in Verdict} == {
            "important", "useless", "referred", "unused"}


class TestSmallReturns:
    def test_span_and_health(self) -> None:
        assert Span(earliest=None).earliest is None
        h = Health(ready=True, failed=False)
        assert h.ready and not h.failed
