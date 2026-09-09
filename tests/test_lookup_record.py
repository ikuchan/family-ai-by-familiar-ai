"""調べものを1つの器にまとめた（環-g・段は）。

1件の調べものについて「どの動作で」「何という語で」が、**6つの入れ物に3通りで**入って
いた。`_inflight`（数）と `_in_flight_lookups`（列）は名前も意味もほぼ同じで、**5箇所で
別々に動かしていた**（`設計方針_ループの語を束ねる` ①）。

| 旧 | 新しい引き方 |
|---|---|
| `_in_flight_lookups` | `result is None` のもの |
| `_lookup_action_by_query` | 語で引いて `.action` |
| `_lookup_index_by_query` | 語で引いて `.index` |
| `_lookup_generation` | 語で引いて `.generation` |
| `_lookup_results` | `result is not None` のもの |
| `_inflight` | `result is None` の数 |

**飛行中の数は導出になる。** 手で揃える箇所が0になり、釣り合いは機械が守る。
"""

from __future__ import annotations

import re
from pathlib import Path

from familiar_agent.loop.event_loop import InformationProcessing, Lookup
from familiar_agent.loop.request import Request

_LOOP = Path(__file__).parent.parent / "src/familiar_agent/loop/event_loop.py"


def _ip() -> InformationProcessing:
    from unittest.mock import MagicMock

    ip = InformationProcessing.__new__(InformationProcessing)
    # `__new__` は `__init__` を通らないので、求めの器は自分で置く（に-5-に-1）。
    ip._req = Request()
    ip._req.lookups = []
    ip._request_generation = 0
    ip._agent = MagicMock()
    return ip


# ── 器 ─────────────────────────────────────────────────────────────────────


def test_one_lookup_holds_everything_about_it():
    lk = Lookup(index=1, action="search_deferred", query="明日の天気", generation=0)
    assert lk.result is None  # 未着は None
    assert lk.in_flight is True


def test_a_lookup_with_a_result_is_no_longer_in_flight():
    lk = Lookup(index=1, action="recall", query="昨日", generation=0, result="見つかった")
    assert lk.in_flight is False


# ── 導出 ───────────────────────────────────────────────────────────────────


def test_the_in_flight_count_is_derived_not_kept():
    """**手で揃えない。** 数は器の列から導く。"""
    ip = _ip()
    ip._req.lookups = [
        Lookup(index=1, action="recall", query="a", generation=0),
        Lookup(index=2, action="recall", query="b", generation=0, result="来た"),
        Lookup(index=3, action="recall", query="c", generation=0),
    ]
    assert ip._in_flight_count == 2


def test_a_lookup_is_found_by_its_query():
    ip = _ip()
    ip._req.lookups = [Lookup(index=7, action="see", query="目の前", generation=3)]
    got = ip._lookup_of("目の前")
    assert (got.index, got.action, got.generation) == (7, "see", 3)
    assert ip._lookup_of("知らない語") is None


def test_the_index_counts_up_within_one_request():
    """通し番号は器の数から決まる（別の変数で数えない）。"""
    ip = _ip()
    assert ip._next_lookup_index() == 1
    ip._req.lookups.append(Lookup(index=1, action="recall", query="a", generation=0))
    assert ip._next_lookup_index() == 2


# ── 旧い入れ物が消えたこと ──────────────────────────────────────────────────


def _code_only() -> str:
    """docstring とコメントを落とした本文。`Lookup` の説明は旧名を語るためである。"""
    import io
    import tokenize

    out: list[str] = []
    with open(_LOOP, "rb") as f:
        for tok in tokenize.tokenize(io.BytesIO(f.read()).readline):
            if tok.type not in (tokenize.COMMENT, tokenize.STRING):
                out.append(tok.string)
    return " ".join(out)


def test_the_six_containers_are_gone():
    src = _code_only()
    for old in (
        "_inflight",
        "_in_flight_lookups",
        "_lookup_action_by_query",
        "_lookup_index_by_query",
        "_lookup_generation",
        "_lookup_results",
        "_lookup_seq",
        "_pending_intent",
    ):
        assert not re.search(rf"\b{old}\b", src), f"{old} が残っている"


def test_nothing_adjusts_the_count_by_hand():
    """`+= 1` / `-= 1` で飛行中の数を動かす箇所が無いこと。"""
    src = _LOOP.read_text(encoding="utf-8")
    assert "_in_flight_count +=" not in src
    assert "_in_flight_count -=" not in src
