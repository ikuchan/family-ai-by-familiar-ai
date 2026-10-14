"""REST が読む専用の計測ログ（記-i・2026-09-13）。

もとは続き先の判定の結末（5 通り）を書く場所として作った。続き先の判定は記-t 段 3（2026-10-15）で撤去し、
いまは調停・申告・層ごとの結果などが書く。ここでは計測ログそのものの約束（置き場所・回転しない・改名のあと作り直す・
`app.log` に流さない）を見る。

計測ログは `~/.cache/familiar-ai/rest_logs/measure.log`（`logs/` の兄弟）。起動時に回転せず、
REST 内省が読み終えたら改名する（記-a-に）。書き手は `WatchedFileHandler` なので、改名の
あとは次の行で新しい file を作り直す。`app.log` には流さない。
"""

from __future__ import annotations

import logging
from pathlib import Path

from familiar_agent.core import measure


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def test_the_measure_log_lives_next_to_logs_and_is_not_rotated_at_startup(tmp_path):
    path = measure.setup(base_dir=tmp_path)
    assert path == tmp_path / "rest_logs" / "measure.log"
    measure.record("調停", 秒=0.4, 分岐="発話")
    measure.record("調停", 秒=1.2, 分岐="完了")
    got = _lines(path)
    assert len(got) == 2
    assert got[0].endswith(" 調停 秒=0.4 分岐=発話")
    assert "秒=1.2 分岐=完了" in got[1]
    # もう一度 setup しても（再起動）、行は消えない。
    measure.setup(base_dir=tmp_path)
    measure.record("調停", 秒=0.1)
    assert len(_lines(path)) == 3


def test_a_renamed_file_is_recreated_on_the_next_line(tmp_path):
    """REST が読み終えて改名したあと、書き手は新しい file を作る（logrotate と同じ約束）。"""
    path = measure.setup(base_dir=tmp_path)
    measure.record("調停", 秒=0.5)
    rotated = path.with_name("measure.log.20260913T2130")
    path.rename(rotated)
    measure.record("調停", 秒=0.9)
    assert len(_lines(rotated)) == 1 and "秒=0.5" in _lines(rotated)[0]
    assert len(_lines(path)) == 1 and "秒=0.9" in _lines(path)[0]


def test_the_measure_logger_does_not_leak_into_the_app_log(tmp_path):
    measure.setup(base_dir=tmp_path)
    assert logging.getLogger(measure.LOGGER_NAME).propagate is False


def test_the_app_wires_the_measure_log_at_startup():
    import inspect

    from familiar_agent import main as main_mod

    src = inspect.getsource(main_mod.setup_logging)
    assert "measure.setup(log_dir)" in src
