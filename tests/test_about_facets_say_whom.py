"""記-ae 段 1：ほかの人について話したことを、その人と話すときに思い出したら、W の行で「誰のことか」が読める。

about の面（話題にされた人の面）の言葉は「[自分のこと] ＋本文」で、その人から見た言葉である。W の行は
「ママが言った: [自分のこと] …」となり、「自分」がパジュか相手か読めなかった。その人の名前で書く。
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from familiar_agent.io.oif import MI, Recalled
from familiar_agent.loop import workspace

WHEN = datetime(2026, 11, 20, 18, 0, tzinfo=timezone.utc)


def _r(obs_id: str, *, key: str, person: str, content: str) -> Recalled:
    return Recalled(
        mi=MI(
            id=f"f-{obs_id}-{person}",
            obs_id=obs_id,
            person_id=person,
            relation_key=key,
            content=content,
            timestamp=WHEN,
            direction="発話",
        ),
        fit=0.8,
        groundedness=0.5,
        confidence=0.9,
    )


def test_an_about_facet_names_whom_it_is_about():
    got = workspace._lines(
        [_r("o1", key="about", person="p-b", content="[自分のこと] 最近忙しそう")],
        {"o1": "ママ"},
        about={"f-o1-p-b": "パパ"},
    )
    assert "ママが言った: [パパのこと] 最近忙しそう" in got


def test_other_facets_are_left_as_they_are():
    got = workspace._lines(
        [_r("o1", key="present", person="p-b", content="[そばに居た] 晩ごはんの話")],
        {"o1": "ママ"},
        about={},
    )
    assert "ママが言った: [そばに居た] 晩ごはんの話" in got


class _Oif:
    def person_names(self, ids):
        return {k: v for k, v in {"p-b": "パパ", "p-c": "こども"}.items() if k in ids}

    def me(self):
        return "パジュ"


def test_about_names_come_from_the_facets_person():
    memories = [
        _r("o1", key="about", person="p-b", content="[自分のこと] 最近忙しそう"),
        _r("o2", key="present", person="p-c", content="[そばに居た] 晩ごはん"),
    ]
    assert workspace.about_names(_Oif(), memories) == {"f-o1-p-b": "パパ"}


def test_one_event_about_two_people_names_each_facet():
    """1 つの出来事に、話題にされた人ぶんの about の面が立つ。面ごとに、その人の名前で書く。"""
    memories = [
        _r("o1", key="about", person="p-b", content="[自分のこと] ふたりとも忙しそう"),
        _r("o1", key="about", person="p-c", content="[自分のこと] ふたりとも忙しそう"),
    ]
    got = workspace._lines(memories, {"o1": "ママ"}, about=workspace.about_names(_Oif(), memories))
    assert "[パパのこと] ふたりとも忙しそう" in got and "[こどものこと] ふたりとも忙しそう" in got


def test_an_about_facet_on_paju_uses_my_name():
    from familiar_agent.person_memory_manager import AGENT_SELF_ID

    memories = [_r("o1", key="about", person=AGENT_SELF_ID, content="[自分のこと] よく話すね")]
    assert workspace.about_names(_Oif(), memories) == {f"f-o1-{AGENT_SELF_ID}": "パジュ"}


def test_a_failed_name_lookup_is_warned_and_keeps_the_tag(caplog):
    class _Broken(_Oif):
        def person_names(self, ids):
            raise RuntimeError("DB が落ちた")

    memories = [_r("o1", key="about", person="p-b", content="[自分のこと] 最近忙しそう")]
    assert workspace.about_names(_Broken(), memories) == {}
    assert any("about の面の名前を引けなかった" in r.getMessage() for r in caplog.records)


def test_the_memory_tool_names_about_facets_too():
    """道具の返りも、W と同じ読み方にする（同じ面が W と道具で食い違わない）。"""
    from familiar_agent.tools.memory import name_about

    assert name_about("[自分のこと] 最近忙しそう", "パパ") == "[パパのこと] 最近忙しそう"
    assert name_about("[そばに居た] 晩ごはん", "パパ") == "[そばに居た] 晩ごはん"


def test_without_a_name_the_line_keeps_its_words():
    """名前が引けなければ書き換えない（名前を捏造しない）。"""
    got = workspace._lines(
        [_r("o1", key="about", person="p-x", content="[自分のこと] 最近忙しそう")],
        {"o1": "ママ"},
        about={},
    )
    assert "[自分のこと] 最近忙しそう" in got


def test_names_by_person_use_the_first_way_of_calling():
    from familiar_agent.tools.memory import ObservationMemory

    memory = SimpleNamespace(
        list_persons=lambda: [
            {"id": "p-b", "name": "パパ、だいこくさま", "display_name": None},
            {"id": "p-c", "name": "ママ", "display_name": "ママ"},
        ]
    )
    memory._displays = lambda: ObservationMemory._displays(memory)
    assert ObservationMemory.person_names(memory, ["p-b", "p-x"]) == {"p-b": "パパ"}
