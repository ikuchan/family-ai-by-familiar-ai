"""会話の中身を、決まったひな形の PDF にして、家のプリンタへ渡す（知-bo 段 1・2026-11-22）。

本人の決定（2026-11-21・一問一答）：刷るのは直前の会話で出た中身（調べものの答え・レシピ・予定の一覧など）で、題・日付・
本文（箇条書き可）を決まったひな形にはめる（画面を見ずに刷るので、形は機械が決める）。白黒・片面。PDF は Qt の書き出しで
作り（新しい依存なし）、CUPS の `lp` で刷る。白黒は IPP の `print-color-mode=monochrome`（前に試した `ColorModel=Gray` は
効かなかった・2026-10-24）、片面は `sides=one-sided`。

PDF を作るには Qt のアプリ（`QGuiApplication` か `QApplication`）が要る（画面のパジュでは動いている。無ければ `PrintError`）。
`send_to_printer` は `lp` を待つ（20 秒まで）ので、非同期の口からは `asyncio.to_thread` で呼ぶ。
"""

from __future__ import annotations

import html
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

#: `lp` を待つ秒。受け付けるだけなので短い（紙が出るのは待たない）。
_LP_TIMEOUT_SEC = 20.0
#: 白黒・片面・A4（本人の決定）。
_LP_OPTIONS = ("print-color-mode=monochrome", "sides=one-sided", "media=A4")
#: 受付番号。`lp` は言語に合わせて言うので（日本語なら「リクエスト ID は …」）、`LC_ALL=C` で呼んで英語で受ける。
_REQUEST_ID = re.compile(r"request id is (\S+)")
_BULLETS = ("- ", "・", "* ")

_STYLE = (
    "body { font-size: 12pt; }"
    " h1 { font-size: 20pt; margin-bottom: 2pt; }"
    " .date { font-size: 10pt; margin-bottom: 14pt; }"
    " li { margin-bottom: 4pt; }"
)


class PrintError(RuntimeError):
    """プリンタが受け付けなかった（わけは `str(e)`）。"""


def page_html(title: str, date_text: str, body: str) -> str:
    """ひな形の HTML。題・日付・本文。本文の「- 」「・」で始まる行は箇条書きにする。言葉は印として読ませない。"""
    parts = [f"<h1>{html.escape(title)}</h1>", f'<p class="date">{html.escape(date_text)}</p>']
    items: list[str] = []

    def flush() -> None:
        if items:
            parts.append("<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
            items.clear()

    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        mark = next((b for b in _BULLETS if line.startswith(b)), None)
        if mark is not None:
            items.append(
                html.escape(line[len(mark) :].strip())
            )  # 印だけを外す（「- -3度」の負の印は残す）
            continue
        flush()
        parts.append(f"<p>{html.escape(line)}</p>")
    flush()
    return f"<html><head><style>{_STYLE}</style></head><body>{''.join(parts)}</body></html>"


def render_pdf(title: str, date_text: str, body: str, path: Path) -> int:
    """ひな形にはめて、A4 の PDF を `path` に書き、ページの数を返す。Qt のアプリが無い、書けなかったときは `PrintError`。"""
    from PySide6.QtCore import QMarginsF, QSizeF
    from PySide6.QtGui import QGuiApplication, QPageLayout, QPageSize, QPdfWriter, QTextDocument

    if QGuiApplication.instance() is None:
        raise PrintError("PDF を作れない：Qt のアプリが動いていない")

    writer = QPdfWriter(str(path))
    writer.setPageLayout(
        QPageLayout(
            QPageSize(QPageSize.PageSizeId.A4),
            QPageLayout.Orientation.Portrait,
            QMarginsF(20, 20, 20, 20),
            QPageLayout.Unit.Millimeter,
        )
    )
    doc = QTextDocument()
    doc.setHtml(page_html(title, date_text, body))
    # 紙の書ける範囲を「点」で渡す（書き出しのピクセルで渡すと、12pt が点ほどの大きさになった・2026-11-22 に描いて確かめた）。
    rect = writer.pageLayout().paintRect(QPageLayout.Unit.Point)
    doc.setPageSize(QSizeF(rect.width(), rect.height()))
    pages = doc.pageCount()
    doc.print_(writer)
    del writer  # 書き終える（ファイルを閉じる）
    try:
        ok = path.read_bytes()[:4] == b"%PDF"
    except OSError:
        ok = False
    if not ok:
        raise PrintError(f"PDF を書けなかった：{path}")
    logger.info("印刷の PDF を作った（%d 字・%d 枚）", len(body), pages)
    return pages


def send_to_printer(
    path: Path,
    *,
    printer: str,
    run: Callable = subprocess.run,
) -> str:
    """`lp` で刷る（白黒・片面・A4）。受付番号を返す。受け付けなければ `PrintError`。"""
    cmd = ["lp", "-d", printer]
    for opt in _LP_OPTIONS:
        cmd += ["-o", opt]
    cmd.append(str(path))
    try:
        r = run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_LP_TIMEOUT_SEC,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise PrintError(f"lp を呼べなかった：{type(e).__name__}") from e
    if r.returncode != 0:
        raise PrintError((r.stderr or r.stdout or "lp が受け付けなかった").strip())
    m = _REQUEST_ID.search(r.stdout or "")
    job = m.group(1) if m else ""
    if job:
        logger.info("印刷を頼んだ：%s（受付 %s）", printer, job)
    else:
        logger.warning("印刷を頼んだが、受付番号を読めなかった：%s", printer)
    return job
