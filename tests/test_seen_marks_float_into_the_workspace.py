"""**見た印は、その求めのあいだ W へ浮かせる**（2026-09-12 実機で露見）。

`see` の帰りは版に載せない（同じ出来事が2件になって W の枠を食う）。中身は `観察` の記録
（見た印）が持つ。`_write_seen_mark` の docstring は「**W へ載せる**」と言っていたが、実際に
やっていたのは `_note_record(obs_id, "見た")`——やりとりの関係と共起の材料に加えるだけで、
W へ浮かせる `open_ids` には入っていなかった。

だから見た印は**似ている順の採点で上位に入ったときしか** W に載らず、「何が見えますか？」に
対して `見えたもの：table、cabinet…` は載らなかった。主LLM は「見たらしいが何が見えたか
書いていない」版を渡され、もう一度 `see` を出し、それが5回続いた。

**新しい欄は作らない。** 「見た」の役割は `turn_records` に既に控えている。
"""

from __future__ import annotations

from familiar_agent.loop import workspace
from familiar_agent.loop.request import Request


def _req(records):
    r = Request()
    r.request_id = "起点"
    r.live_version_id = "版2"
    r.turn_records = list(records)
    return r


def test_seen_marks_of_this_request_are_open():
    r = _req([("起点", "起点"), ("版1", "版"), ("見た1", "見た"), ("版2", "版")])
    assert workspace.open_ids(r) == ["起点", "版2", "見た1"]


def test_other_roles_do_not_float():
    """版・つなぎ・答えは浮かせない。版は生きている1つだけが載る。"""
    r = _req([("起点", "起点"), ("版1", "版"), ("つなぎ1", "つなぎ"), ("答え1", "答え")])
    assert workspace.open_ids(r) == ["起点", "版2"]


def test_seen_marks_of_a_closed_exchange_do_not_float():
    """前のやりとりで見たものは、次の求めには載せない（`exchange_start` より前）。"""
    r = _req([("見た0", "見た"), ("起点", "起点"), ("見た1", "見た")])
    r.exchange_start = 1
    assert workspace.open_ids(r) == ["起点", "版2", "見た1"]


def test_the_docstring_and_the_code_agree():
    """「W へ載せる」と言っている以上、載る。"""
    import inspect

    from familiar_agent.loop.event_loop import InformationProcessing

    doc = inspect.getdoc(InformationProcessing._write_seen_mark) or ""
    assert "W へ" in doc
    assert "見た" in inspect.getsource(workspace.open_ids)
