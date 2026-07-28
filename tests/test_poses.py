"""定点（見回り・在席マップ・norm が共有する N 個の向き）。

`知覚在席` §3-3 が定めるとおり、PTZ カメラは一度に1方向しか見ないので、在席も「普通」も
**定点ごと**に持つ。現在の向きは最寄りの定点へ対応づけ、どの定点からも離れていれば
「移動中」として在席と norm を更新しない（振動中ゲート）。

定点は**カメラのプリセットと Config の和**である。実機のプリセットは1件しか無く、部屋の
右半分はテレビが塞いでいて使えないため、Config が主でプリセットが補いになる。人が後から
カメラのアプリでプリセットを足したときに、Config から消す手作業が要らないよう、しきい値内に
重なる向きは1つに畳む。

値は ONVIF の正規化座標 $[-1, 1]$ で、角度ではない。実機で確かめた可動範囲は
pan・tilt とも $[-1, 1]$。
"""

from __future__ import annotations

from familiar_agent.poses import Pose, merge_poses, nearest_pose, parse_poses

# 実機で決めた3点（2026-07-28）。右半分はテレビで使えない。
_REAL = [
    Pose("窓側", 0.000, -0.50),
    Pose("出入り口", -0.129, -0.50),
    Pose("襖側", -0.667, -0.29),
]
_TOL = 0.05


# --- 読み込み -------------------------------------------------------------


def test_poses_are_read_from_a_single_line():
    got = parse_poses("窓側:0.000,-0.50;出入り口:-0.129,-0.50;襖側:-0.667,-0.29")
    assert got == _REAL


def test_spaces_around_the_separators_are_ignored():
    got = parse_poses(" 窓側 : 0.0 , -0.5 ; 出入り口:-0.129,-0.5 ")
    assert [p.name for p in got] == ["窓側", "出入り口"]


def test_an_empty_setting_yields_no_poses():
    assert parse_poses("") == []
    assert parse_poses("   ") == []


def test_a_malformed_entry_is_dropped_without_killing_the_rest():
    # 設定の誤記でカメラごと止めない。読めたものだけ使う。
    got = parse_poses("窓側:0.0,-0.5;こわれた;もっと:こわれた,x;出入り口:-0.129,-0.5")
    assert [p.name for p in got] == ["窓側", "出入り口"]


def test_a_pose_outside_the_reachable_range_is_dropped():
    # 可動範囲は pan・tilt とも [-1,1]（実機で確認）。外れた値は届かない。
    assert parse_poses("外:1.5,0.0") == []
    assert parse_poses("外:0.0,-2.0") == []


# --- プリセットとの統合 ---------------------------------------------------


def test_a_preset_that_overlaps_a_configured_pose_is_folded_into_it():
    """実機のプリセット「出入り口」は pan=-0.1294 で、Config の -0.129 と 0.0004 しか違わない。

    別々の定点として数えると、同じ向きに2つの「普通」と2つの在席マップができる。
    """
    presets = [Pose("出入り口", -0.1294, -0.50)]
    got = merge_poses(_REAL, presets, _TOL)
    assert len(got) == 3


def test_the_configured_name_wins_over_the_preset_name():
    presets = [Pose("玄関", -0.1294, -0.50)]
    got = merge_poses(_REAL, presets, _TOL)
    assert "玄関" not in [p.name for p in got]
    assert "出入り口" in [p.name for p in got]


def test_a_preset_somewhere_new_becomes_a_pose():
    # 人がカメラのアプリでプリセットを足したら、そのまま定点が増える。
    presets = [Pose("台所", 0.8, -0.4)]
    got = merge_poses(_REAL, presets, _TOL)
    assert len(got) == 4 and "台所" in [p.name for p in got]


def test_presets_overlapping_each_other_are_folded_too():
    presets = [Pose("あ", 0.8, -0.4), Pose("い", 0.81, -0.4)]
    assert len(merge_poses([], presets, _TOL)) == 1


# --- 最寄り判定 -----------------------------------------------------------


def test_the_nearest_pose_is_found_when_the_camera_sits_on_one():
    assert nearest_pose(_REAL, -0.129, -0.50, _TOL).name == "出入り口"


def test_a_small_drift_still_counts_as_the_same_pose():
    # 絶対移動には誤差がある。少しずれただけで「移動中」にすると、norm が育たない。
    assert nearest_pose(_REAL, -0.140, -0.505, _TOL).name == "出入り口"


def test_being_between_poses_means_moving():
    # どの定点からも離れている＝移動中。在席も「普通」も更新しない（振動中ゲート）。
    assert nearest_pose(_REAL, -0.35, -0.50, _TOL) is None


def test_the_three_real_poses_are_never_confused():
    # いちばん近い2点の差は 0.129 で、しきい値 0.05 の2倍を超える。
    for p in _REAL:
        assert nearest_pose(_REAL, p.pan, p.tilt, _TOL).name == p.name


def test_no_poses_means_never_on_one():
    assert nearest_pose([], 0.0, 0.0, _TOL) is None
