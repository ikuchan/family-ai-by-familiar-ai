"""計測ログの読み手と集計（記-a-に・2026-09-14）。数字だけを返す（生の行は LLM に渡さない）。"""

from __future__ import annotations

from pathlib import Path

from familiar_agent.core import measure


def _write(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "rest_logs" / "measure.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_rows_are_parsed_by_kind_into_key_value_dicts(tmp_path):
    _write(
        tmp_path,
        [
            "2026-09-14T10:00:00 調停 秒=0.91 分岐=light 時間切れ=no",
            "2026-09-14T10:00:05 続き先 結末=続き 相手=abc 起点=def",
            "2026-09-14T10:00:06 気分 P=0.12 Pn=0.10 A=0.50 Dom=0.51",
        ],
    )
    rows = measure.read_rows(base_dir=tmp_path)
    assert [r.kind for r in rows] == ["調停", "続き先", "気分"]
    assert rows[0].fields == {"秒": "0.91", "分岐": "light", "時間切れ": "no"}
    assert rows[0].when.hour == 10


def test_arbiter_summary_gives_counts_quantiles_and_timeout_rate(tmp_path):
    lines = [
        f"2026-09-14T10:00:{i:02d} 調停 秒={s} 分岐=full 時間切れ={'yes' if s >= 5 else 'no'}"
        for i, s in enumerate([0.8, 0.9, 1.0, 1.2, 5.0])
    ]
    _write(tmp_path, lines)
    s = measure.summarize_arbiter(measure.read_rows(base_dir=tmp_path))
    assert s["件数"] == 5 and s["時間切れ"] == 1 and abs(s["時間切れの割合"] - 0.2) < 1e-9
    assert s["中央"] == 1.0 and s["p90"] >= 1.2 and s["最大"] == 5.0


def test_window_summary_matches_follow_targets_against_the_edge_and_beyond(tmp_path):
    _write(
        tmp_path,
        [
            "2026-09-14T10:00:00 直近 窓=3 端=q2 外=q1",
            "2026-09-14T10:00:01 続き先 結末=続き 相手=q2 起点=x1",  # 端を参照
            "2026-09-14T10:01:00 直近 窓=3 端=q3 外=q2",
            "2026-09-14T10:01:01 続き先 結末=続き 相手=q2 起点=x2",  # 窓の外を参照
            "2026-09-14T10:02:00 直近 窓=3 端=q4 外=q3",
            "2026-09-14T10:02:01 続き先 結末=途切れ 相手=- 起点=x3",
            "2026-09-14T10:03:00 直近 窓=3 端=q5 外=q4",
            "2026-09-14T10:03:01 続き先 結末=続き 相手=q5 起点=x4",  # 端を参照
        ],
    )
    s = measure.summarize_window(measure.read_rows(base_dir=tmp_path))
    assert s["続き"] == 3 and s["端を参照"] == 2 and s["外を参照"] == 1
    assert abs(s["端か外の割合"] - 1.0) < 1e-9


def test_inner_state_summary_gives_quantiles_per_axis(tmp_path):
    lines = [
        f"2026-09-14T10:{i:02d}:00 気分 P={0.1 + i * 0.01:.2f} Pn=0.10 A=0.50 Dom=0.50"
        for i in range(20)
    ]
    _write(tmp_path, lines)
    s = measure.summarize_inner_state(measure.read_rows(base_dir=tmp_path))
    assert set(s) >= {"P", "Pn", "A", "Dom"}
    assert s["P"]["p10"] < s["P"]["p90"] and s["Pn"]["p10"] == s["Pn"]["p90"] == 0.10


def test_relation_summary_counts_references_by_order(tmp_path):
    _write(
        tmp_path,
        [
            "2026-09-14T10:00:00 関連 遠い=a,b 掘り=c,d",
            "2026-09-14T10:00:01 申告 important=a useless=- referred=d unused=b,c",
            "2026-09-14T10:01:00 関連 遠い=e 掘り=f",
            "2026-09-14T10:01:01 申告 important=- useless=- referred=f unused=e",
        ],
    )
    s = measure.summarize_relation(measure.read_rows(base_dir=tmp_path))
    assert s == {"遠い": 1, "掘り": 2, "件数": 2}  # 件数＝突き合わせた申告の行（記-k）


def test_rotate_renames_the_file_with_a_timestamp(tmp_path):
    p = _write(tmp_path, ["2026-09-14T10:00:00 調停 秒=0.9 分岐=light 時間切れ=no"])
    rotated = measure.rotate(base_dir=tmp_path)
    assert (
        rotated is not None
        and rotated.parent == p.parent
        and rotated.name.startswith("measure.log.")
    )
    assert not p.exists() and rotated.exists()
    assert measure.rotate(base_dir=tmp_path) is None  # 無ければ何もしない
