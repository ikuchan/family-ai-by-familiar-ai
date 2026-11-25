"""記-ah 段 3：思い出した記録のうち、置き換えられた事実に期間を添えて W に載せる（`設計方針_事実の移り変わり` §3 の c）。

置き換えられた事実も思い出し、点は下げない。W の 1 行の末尾に「（「話題」は いつ に置き換わった・新しい記録 id:…）」を添え、
古いことを今のこととして言わないようにする。置き換えの関係は、W を組むときに思い出した記録の id でまとめて読む。
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from familiar_agent.io.oif import MI, Recalled
from familiar_agent.loop import workspace
from familiar_agent.store.relations import Replacement

WHEN = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)


def _r(obs_id: str, content: str) -> Recalled:
    return Recalled(
        mi=MI(
            id=f"f-{obs_id}",
            obs_id=obs_id,
            person_id="p-mama",
            relation_key="actor",
            content=content,
            timestamp=WHEN,
            direction="発話",
        ),
        fit=0.8,
        groundedness=0.5,
        confidence=0.9,
    )


_REP = Replacement(
    old_key=10,
    old_sentence="さぶろうの習い事",
    new_obs="0123456789abcdef-new",
    new_key=20,
    new_sentence="さぶろうの習い事",
    valid_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
)


def test_a_replaced_fact_is_shown_with_when_it_changed():
    got = workspace._lines(
        [_r("o1", "さぶろうはサッカーに通ってる")],
        {"o1": "ママ"},
        replaced={"o1": [_REP]},
    )
    assert "さぶろうはサッカーに通ってる" in got  # 思い出すのは変えない
    assert "（「さぶろうの習い事」は 2026-10-01 に置き換わった・新しい記録 id:0123456789ab）" in got


def test_a_record_without_replacements_has_no_note():
    """**反証の側**：置き換えられていない記録には何も添えない。"""
    got = workspace._lines([_r("o2", "明日は晴れ")], {"o2": "ママ"}, replaced={"o1": [_REP]})
    assert "置き換わった" not in got


def test_the_replacements_are_read_once_for_the_shown_records():
    calls: list = []

    def replacements_of(ids):
        calls.append(list(ids))
        return {"o1": [_REP]}

    oif = SimpleNamespace(replacements_of=replacements_of)
    got = workspace.replaced_facts(oif, [_r("o1", "a"), _r("o2", "b")])
    assert calls == [["o1", "o2"]] and list(got) == ["o1"]


def test_a_failed_read_is_warned_and_shows_nothing(caplog):
    def broken(ids):
        raise RuntimeError("DB が落ちた")

    got = workspace.replaced_facts(SimpleNamespace(replacements_of=broken), [_r("o1", "a")])
    assert got == {}
    assert any("置き換え" in r.getMessage() for r in caplog.records)
