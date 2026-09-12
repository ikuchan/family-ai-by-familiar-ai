"""**open な記録は、W の枠を予約して必ず載る**（2026-09-12 実機で露見）。

`open_ids` は「活性に下限を課して W へ浮かせる」と言っていたが、実際にやっていたのは
**候補に入った記録の根づきを底上げする**ことだけで、候補にすら入らなければ効かず、入っても
採点で負ければ載らなかった。

実機では、手がかりが版の文面なので**他の版が「似ている」として上位7件を独占**し、
この求めで見たもの（`見えたもの：person、table、chair…`）は候補に在っても W に載らなかった。
主LLM は「見たらしいが何が見えたか書いていない」版を渡され、`see` を5回出した。

open は「この求めのために書いた、まだ決着していない記録」である。**似ている順で競わせる
ものではない。** 枠を予約して必ず載せ、残りの枠を採点で埋める。
"""

from __future__ import annotations


from familiar_agent.tools.memory import _seat_open_records


def _r(i, fit):
    return {"memory_id": i, "fit": fit}


def test_open_records_are_seated_first_then_the_rest_by_score():
    results = [_r("v1", 0.9), _r("v2", 0.8), _r("v3", 0.7), _r("seen", 0.1), _r("v4", 0.6)]
    out = _seat_open_records(results, open_ids={"seen"}, n=3)
    assert [r["memory_id"] for r in out] == ["seen", "v1", "v2"]


def test_without_open_records_the_order_is_by_score():
    results = [_r("a", 0.5), _r("b", 0.9)]
    out = _seat_open_records(results, open_ids=set(), n=2)
    assert [r["memory_id"] for r in out] == ["b", "a"]


def test_open_records_do_not_exceed_the_frame():
    """open が枠より多くても、枠は超えない（open どうしは採点順）。"""
    results = [_r("o1", 0.2), _r("o2", 0.9), _r("o3", 0.5), _r("x", 0.99)]
    out = _seat_open_records(results, open_ids={"o1", "o2", "o3"}, n=2)
    assert [r["memory_id"] for r in out] == ["o2", "o3"]


def test_an_open_record_that_is_not_a_candidate_cannot_be_seated():
    """候補に無いものは載せられない（一次絞りの側の話）。黙って増やさない。"""
    results = [_r("a", 0.5)]
    out = _seat_open_records(results, open_ids={"ghost"}, n=3)
    assert [r["memory_id"] for r in out] == ["a"]


def test_the_floor_does_not_drop_open_records():
    """**床の手前で消えたら席は要らない。** 実機で見た印が fit=0.010、床 0.05 で落ちた。"""
    import inspect

    from familiar_agent.tools import memory

    assert 'final < min_score and row["id"] not in _open' in inspect.getsource(memory)
