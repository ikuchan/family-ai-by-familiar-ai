"""計測ログ——REST 内省が読む専用の記録（記-i・2026-09-13）。

`app.log` は起動ごとに `logs/` へ回転し、人が読むためのものである。ここは**機械（REST 内省・
記-a-に）が数えるための行**を溜める別の file で、`logs/` の兄弟 `rest_logs/measure.log` に置く。
起動時に回転しない。REST が読み終えたら同じ dir 内で `measure.log.<読んだ時刻>` へ改名する
（回転するのは REST）。書き手は `WatchedFileHandler` なので、改名の次の行で新しい file を作る
（logrotate と同じ約束）。`app.log` には流さない（`propagate=False`）。

行は `時刻 種別 key=value …` の 1 行。機械が拾いやすく、人が見ても読める。値に空白と
改行は入れない。累計は持たない——並びのまま渡せば傾向が出る（累計値は傾向を渡せない）。
"""

from __future__ import annotations

import logging
from logging.handlers import WatchedFileHandler
from pathlib import Path

LOGGER_NAME = "familiar_agent.measure"
DIR_NAME = "rest_logs"
FILE_NAME = "measure.log"

_logger = logging.getLogger(LOGGER_NAME)


def default_base_dir() -> Path:
    return Path.home() / ".cache" / "familiar-ai"


def setup(base_dir: "Path | None" = None) -> Path:
    """計測ログの書き手を 1 つだけ付け、file の場所を返す（`main.setup_logging` から 1 回）。"""
    path = (base_dir or default_base_dir()) / DIR_NAME / FILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    for h in list(_logger.handlers):
        _logger.removeHandler(h)
        h.close()
    handler = WatchedFileHandler(str(path), encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"))
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    return path


def record(kind: str, **fields: object) -> None:
    """1 行書く。`kind` は種別（`続き先` など）、続きは `key=value` の並び（渡した順）。"""
    if not _logger.handlers:
        return  # 書き手が無い（試験や CLI の一部）。黙って捨てる——計測は主の仕事ではない
    body = " ".join(f"{k}={_clean(v)}" for k, v in fields.items())
    _logger.info("%s %s", kind, body)


def _clean(value: object) -> str:
    text = str(value if value is not None else "-")
    return text.replace("\n", " ").replace(" ", "_") or "-"
