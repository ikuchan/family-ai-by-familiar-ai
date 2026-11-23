"""定点ごとの前の見えの保存層（`pose_views`・知-i 段 2）。

前の見えは、見比べて「物が変わった」と判定したとき（と、最初に落ち着いた見えを見たとき）だけ置き換える。
「明るさだけ」「変わっていない」と判定したときは、見比べた時刻だけを残す（同じ定点を続けて見比べない）。
"""

from __future__ import annotations

import time

from familiar_agent.store.pose_views import PoseViewStore

_DIM = 384


def _vec(first: float) -> list[float]:
    v = [0.0] * _DIM
    v[0] = first
    return v


def _store():
    from familiar_agent.db import get_db

    return PoseViewStore(get_db().conn())


def test_an_unknown_pose_has_no_earlier_view():
    assert _store().load("まだ見ていない定点") is None


def test_a_saved_view_comes_back_with_its_picture():
    s = _store()
    s.save("窓側", _vec(1.0), b"\xff\xd8JPEG")
    got = s.load("窓側")
    assert got is not None
    assert abs(got.embedding[0] - 1.0) < 1e-6 and got.image == b"\xff\xd8JPEG"
    # 置き換えた時は見比べた時でもある（続けて見比べない）
    assert got.checked_at is not None and abs(got.checked_at - time.time()) < 60


def test_saving_again_replaces_the_view():
    s = _store()
    s.save("押入れ", _vec(1.0), b"a")
    s.save("押入れ", _vec(0.5), b"b")
    got = s.load("押入れ")
    assert got is not None and got.image == b"b" and abs(got.embedding[0] - 0.5) < 1e-6


def test_marking_checked_keeps_the_view_and_records_the_time():
    s = _store()
    s.save("正面", _vec(1.0), b"a")
    with s._conn.cursor() as cur:  # 見比べた時刻を 1 日前にしてから印を付ける
        cur.execute(
            "UPDATE pose_views SET checked_at = now() - interval '1 day' WHERE pose_name = '正面'"
        )
    s._conn.commit()
    s.mark_checked("正面")
    got = s.load("正面")
    assert got is not None and got.image == b"a"
    assert got.checked_at is not None and abs(got.checked_at - time.time()) < 60
