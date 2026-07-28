"""定点ごとの「見えの普通」の保存層（`pose_norms`）。

行数は定点の数しかない（実機で3）。定点名で1件ずつ引き、EMA で上書きする。部屋の映像から
作る値なので `agent_state` には混ぜず、専用のテーブルへ置く。
"""

from __future__ import annotations

import pytest

from familiar_agent.store.pose_norms import PoseNormStore

_DIM = 384


def _vec(first: float) -> list[float]:
    v = [0.0] * _DIM
    v[0] = first
    return v


def test_an_unknown_pose_has_no_norm_yet():
    from familiar_agent.db import get_db

    s = PoseNormStore(get_db().conn())
    assert s.load("まだ見ていない定点") == (None, 0)


def test_a_saved_norm_comes_back():
    from familiar_agent.db import get_db

    s = PoseNormStore(get_db().conn())
    s.save("窓側", _vec(1.0), observations=1)
    norm, n = s.load("窓側")
    assert n == 1
    assert norm is not None and abs(norm[0] - 1.0) < 1e-6


def test_saving_again_replaces_the_norm_rather_than_adding_a_row():
    from familiar_agent.db import get_db

    s = PoseNormStore(get_db().conn())
    s.save("出入り口", _vec(1.0), observations=1)
    s.save("出入り口", _vec(0.5), observations=2)
    norm, n = s.load("出入り口")
    assert n == 2
    assert norm is not None and abs(norm[0] - 0.5) < 1e-6


def test_each_pose_keeps_its_own_norm():
    from familiar_agent.db import get_db

    s = PoseNormStore(get_db().conn())
    s.save("襖側", _vec(1.0), observations=3)
    s.save("窓側2", _vec(-1.0), observations=7)
    assert s.load("襖側")[1] == 3
    assert s.load("窓側2")[1] == 7


def test_a_wrong_sized_vector_is_refused_rather_than_stored():
    # 別のモデルへ替えたときに、次元の違う値が混ざると距離が意味を失う。
    from familiar_agent.db import get_db

    s = PoseNormStore(get_db().conn())
    with pytest.raises(ValueError):
        s.save("窓側3", [0.0, 1.0], observations=1)
