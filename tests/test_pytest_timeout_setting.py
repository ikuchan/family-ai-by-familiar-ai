"""テストの打ち切り時間。

止まったテストを待ち続けないよう `pytest-timeout` を入れている（一式が 11 分たっても
終わらない実行が1度あり、どのテストで止まっているのか分からなかったため）。

**60 秒は短すぎた。** 実機の GUI やカメラや GPU と同時に回すと一式が 50 秒→100 秒に伸び、
その負荷でワーカーが打ち切りに掛かって落ちた（5回中3回）。何も動かさずに回すと10回とも
緑なので、**本物の停止ではなく誤射**である。

最も遅い1件は 13.2 秒（実測）。120 秒なら9倍の余裕があり、本当に止まったテストは変わらず
捕まえられる。
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def _pytest_config() -> dict:
    root = Path(__file__).resolve().parent.parent
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return data["tool"]["pytest"]["ini_options"]


def test_a_test_is_cut_off_after_two_minutes():
    assert _pytest_config()["timeout"] == 120


def test_the_cutoff_can_stop_blocking_calls():
    # thread：DB 待ちやネットワーク待ちのようにブロックしている呼び出しも止められる。
    # signal では Python へ制御が戻るまで効かない。
    assert _pytest_config()["timeout_method"] == "thread"
