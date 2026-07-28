"""定点別 presence マップ（在/不在の層）。

`知覚在席` §3-3：PTZ カメラは一度に1方向しか見ないので、**単一フレームの不検出は部屋が
空であることを意味しない**。在席は定点ごとの最終在席時刻として持ち、部屋レベルの「誰か
居る」は滞留窓での集約とする。空判定は、関連する定点を見て回って窓内に誰も居ないときだけ。

この層は**誰かを問わない**。顔の照合（#17）が無くても動く。滞留窓 120 秒は `課題5` §I の
確定値で、`person_memory_manager.py` が持っていた値と同じ。
"""

from __future__ import annotations

from familiar_agent.presence_map import PresenceMap

_WINDOW = 120.0
_POSES = ["窓側", "出入り口", "襖側"]


def _map():
    return PresenceMap(_POSES, window_sec=_WINDOW)


# --- 見た・見なかった -----------------------------------------------------


def test_a_fresh_map_is_not_occupied():
    assert _map().room_occupied(now=1000.0) is False


def test_seeing_someone_makes_the_room_occupied():
    m = _map()
    m.mark_seen("出入り口", at=1000.0)
    assert m.room_occupied(now=1000.0) is True


def test_presence_lasts_for_the_whole_window():
    m = _map()
    m.mark_seen("出入り口", at=1000.0)
    assert m.room_occupied(now=1000.0 + _WINDOW - 1) is True
    assert m.room_occupied(now=1000.0 + _WINDOW + 1) is False


def test_looking_without_finding_anyone_does_not_create_presence():
    m = _map()
    m.mark_checked("出入り口", at=1000.0)
    assert m.room_occupied(now=1000.0) is False


def test_one_empty_pose_does_not_empty_the_room():
    """ここが `知覚在席` §3-3 の核心である。

    出入り口に人が居るのに、襖側を見て誰も居なかったからといって部屋が空になっては困る。
    """
    m = _map()
    m.mark_seen("出入り口", at=1000.0)
    m.mark_checked("襖側", at=1001.0)
    assert m.room_occupied(now=1002.0) is True


def test_an_unknown_pose_is_ignored_rather_than_added():
    # 定点は一覧で決まる。知らない名前で穴が空くと、見回りの巡回対象とずれる。
    m = _map()
    m.mark_seen("台所", at=1000.0)
    assert m.room_occupied(now=1000.0) is False


# --- どこに居るか ---------------------------------------------------------


def test_the_poses_where_someone_was_seen_are_listed():
    m = _map()
    m.mark_seen("出入り口", at=1000.0)
    m.mark_seen("窓側", at=1010.0)
    assert set(m.poses_seen(now=1020.0)) == {"出入り口", "窓側"}


def test_a_pose_drops_off_the_list_after_the_window():
    m = _map()
    m.mark_seen("出入り口", at=1000.0)
    assert m.poses_seen(now=1000.0 + _WINDOW + 1) == []


# --- 見回りが次に見る先 ---------------------------------------------------


def test_the_never_looked_pose_comes_first():
    """見回り（S5）は「最も薄れた定点」を選ぶ。一度も見ていない定点が最優先である。"""
    m = _map()
    m.mark_checked("出入り口", at=1000.0)
    m.mark_checked("窓側", at=1001.0)
    assert m.stalest_pose(now=1002.0) == "襖側"


def test_the_longest_unchecked_pose_comes_first():
    m = _map()
    m.mark_checked("襖側", at=900.0)
    m.mark_checked("出入り口", at=1000.0)
    m.mark_checked("窓側", at=1001.0)
    assert m.stalest_pose(now=1002.0) == "襖側"


def test_seeing_someone_also_counts_as_having_looked():
    # 人が見つかったなら、その定点は「見た」でもある。両方を別々に記録すると、
    # 人が居続ける定点ばかり「まだ見ていない」ことになって巡回が偏る。
    m = _map()
    m.mark_seen("襖側", at=1000.0)
    m.mark_checked("出入り口", at=900.0)
    m.mark_checked("窓側", at=901.0)
    assert m.stalest_pose(now=1002.0) == "出入り口"


def test_no_poses_means_nothing_to_visit():
    assert PresenceMap([], window_sec=_WINDOW).stalest_pose(now=1.0) is None
