"""パジュの状態図の窓（環-bb 三段目・2026-11-21）。

本人の決定 (4)（2026-11-17）「パジュの画面に状態図の窓を足し、いまいる状態を光らせ、最近の移りを時刻と事象の名前で並べる」。
図は `loop/paju_state_chart` がコードから作り、リポジトリに置いた mermaid.js（11.4.1・`assets/mermaid/`・2026-11-21 本人の
承認）が描く。ネットは使わない。いまの状態は動いている状態図から直に読み、最近の移りはログの末尾の「状態図 … で移った」の行から
読む（窓は印を持たない）。描き直すのは、いまの状態かログが変わったときだけ。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QListWidget, QVBoxLayout, QWidget

from .loop import paju_state_chart as chart

logger = logging.getLogger(__name__)

_ASSETS = Path(__file__).resolve().parent / "assets" / "mermaid"
#: 最近の移りに並べる数と、読むログの末尾の大きさ（移りの 1 行は 200 字ほど）。
RECENT = 20
_TAIL_BYTES = 64 * 1024

_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
body { margin: 0; background: #fafafa; font-family: sans-serif; }
#g { padding: 8px; } #err { color: #b00020; white-space: pre-wrap; padding: 8px; }
</style></head><body><div id="g"></div><div id="err"></div>
<script src="mermaid.min.js"></script><script>
mermaid.initialize({ startOnLoad: false, securityLevel: "strict", state: { useMaxWidth: false } });
let n = 0;
function draw(text) {
  n += 1;
  mermaid.render("g" + n, text).then(r => {
    document.getElementById("g").innerHTML = r.svg;
    document.getElementById("err").textContent = "";
  }).catch(e => { document.getElementById("err").textContent = "図を描けなかった：" + e; });
}
</script></body></html>"""


def log_path() -> "Path | None":
    """いま書いているログのファイル（根のロガーのファイル出力）。無ければ None。"""
    for h in logging.getLogger().handlers:
        name = getattr(h, "baseFilename", None)
        if isinstance(h, logging.FileHandler) and name:
            return Path(name)
    return None


def recent_transitions(path: "Path | None", limit: int = RECENT) -> "list[tuple[str, str, str]]":
    """ログの末尾から、最近の移りを新しいものから `limit` 個。読めなければ空。"""
    if path is None:
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - _TAIL_BYTES))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return list(reversed(chart.transitions_in(text.splitlines())))[:limit]


class StateChartWindow(QWidget):
    """上に状態図（いまの状態を光らせる）、下に最近の移り（時刻と事象の名前）を並べる窓。"""

    def __init__(self, parent: "QWidget | None" = None) -> None:
        super().__init__(parent)
        from PySide6.QtWebEngineWidgets import QWebEngineView

        self.setWindowTitle("パジュの状態図")
        self.resize(900, 900)
        self._view = QWebEngineView(self)
        self._recent = QListWidget(self)
        self._recent.setMaximumHeight(220)
        layout = QVBoxLayout(self)
        layout.addWidget(self._view, stretch=4)
        layout.addWidget(self._recent, stretch=1)
        self._ready = False
        self._shown: "frozenset[str] | None" = None
        self._log_size = -1
        self._listed: "list[tuple[str, str, str]]" = []
        self._view.loadFinished.connect(self._on_loaded)
        self._view.setHtml(_HTML, QUrl.fromLocalFile(str(_ASSETS) + "/"))

    def _on_loaded(self, ok: bool) -> None:
        self._ready = bool(ok)
        self._shown = None  # 読み直したら描き直す
        if not ok:
            logger.warning("状態図の窓を読み込めなかった")

    def refresh(self, sm: Any) -> None:
        """いまの状態が変わっていれば描き直し、移りが増えていれば最近の移りを並べ直す。画面の 350 ミリ秒の輪から呼ぶ。

        状態図がまだ無い（起動の途中）ときは、光らせない図を描く。
        """
        if self._ready:
            now: "frozenset[str] | None" = frozenset()
            if sm is not None:
                try:
                    now = frozenset(chart.active_names(sm))
                except Exception:  # noqa: BLE001
                    now = None
            if now is not None and now != self._shown:
                self._shown = now
                self._view.page().runJavaScript(f"draw({json.dumps(chart.highlighted(now))})")
        path = log_path()
        try:
            size = path.stat().st_size if path is not None else -1
        except OSError:  # 日が変わってログを入れ替えた一瞬など
            size = -1
        if size == self._log_size:
            return
        self._log_size = size
        recent = recent_transitions(path)
        if recent != self._listed:  # 移りが増えたときだけ並べ直す（見ている場所を毎回戻さない）
            self._listed = recent
            self._recent.clear()
            for at, event, _desc in recent:
                self._recent.addItem(f"{at[11:]}  {event}")
