"""引き方の 1 行が、載った件数を正しく言う（出-ai・2026-09-22）。

実機 15:48、`[この想起：いまの相手の面・7 件まで（11 件）・…]` と出ていた。「上限 7 なのに
11 載っている」と読めるが、**11 はすべて W に載っている**——採点で採った 7 件に、関連想起
（拡散・`diffuse_max_add`＝4）が足した 4 件が加わった数である。`Workspace.build` は件数を
切らない。

課題に書いた見立て（「W に載るのは枠に入るぶんだけ」）は事実と違った。直すのは**書き方**で、
採点ぶんと関連ぶんを分けて書く。分けるのは、引き直す道具（`recall_deeper` ほか）が効くのは
採点ぶんのほうだからである。
"""

from __future__ import annotations

from familiar_agent.config import MemoryConfig
from familiar_agent.io.oif import Recalled
from familiar_agent.loop.workspace import describe_basis


def _rec(by_association: bool) -> Recalled:
    from familiar_agent.io.oif import MI

    return Recalled(
        mi=MI(
            id="m",
            obs_id="o",
            person_id="",
            relation_key="",
            content="x",
            timestamp=None,
            direction="",
        ),
        fit=0.5,
        groundedness=0.5,
        by_association=by_association,
    )


def _basis(scored: int, associated: int) -> str:
    memories = [_rec(False) for _ in range(scored)] + [_rec(True) for _ in range(associated)]
    return describe_basis(MemoryConfig(), viewpoint="p1", time_ref=None, memories=memories)


def test_the_two_kinds_are_counted_apart():
    got = _basis(7, 4)
    assert "似ている順に 7 件" in got
    assert "関連で 4 件" in got


def test_nothing_from_association_says_only_the_scored_count():
    """関連がゼロなら、その節は出さない。無い数を 0 と書いても読む手がかりにならない。"""
    got = _basis(5, 0)
    assert "似ている順に 5 件" in got
    assert "関連" not in got


def test_an_empty_recall_still_reads():
    got = _basis(0, 0)
    assert "似ている順に 0 件" in got


def test_the_other_conditions_are_unchanged():
    """面・基準・直近の窓はそのまま（出-ah の 1 行の役目）。"""
    got = _basis(7, 4)
    assert "いまの相手の面" in got
    assert "いま基準" in got
    assert "直近 5 分" in got


def test_the_cap_is_not_shown_as_a_total():
    """`7 件まで（11 件）` の形はやめる。上限より多い数が載ったように読める。"""
    assert "件まで" not in _basis(7, 4)


# ── 由来が Recalled に残る ────────────────────────────────────────────────


def test_a_diffuse_row_is_marked():
    from familiar_agent.io.oif import _to_recalled

    got = _to_recalled({"memory_id": "o", "summary": "x", "retrieval_method": "diffuse"})
    assert got.by_association is True


def test_a_scored_row_is_not_marked():
    from familiar_agent.io.oif import _to_recalled

    got = _to_recalled({"memory_id": "o", "summary": "x", "retrieval_method": "semantic"})
    assert got.by_association is False


def test_a_row_without_the_field_is_not_marked():
    from familiar_agent.io.oif import _to_recalled

    assert _to_recalled({"memory_id": "o", "summary": "x"}).by_association is False
