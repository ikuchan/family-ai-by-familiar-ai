"""知-bo 段 1：会話の中身を、決まったひな形の PDF にして、家のプリンタへ白黒・片面で渡す（2026-11-22）。

本人の決定（2026-11-21）：刷るのは題・日付・本文（箇条書き可）。白黒は IPP の `print-color-mode=monochrome`（前に試した
`ColorModel=Gray` は効かなかった）、片面は `sides=one-sided`。PDF は Qt で作る（新しい依存なし）。本物の紙は刷らない（`lp` は偽物）。
"""

from __future__ import annotations

import shutil
import subprocess
import unicodedata
from types import SimpleNamespace

import pytest

from familiar_agent.io import printer

QApplication = pytest.importorskip("PySide6.QtWidgets").QApplication


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def test_the_pdf_is_an_a4_page_with_the_words(qapp, tmp_path):
    path = tmp_path / "out.pdf"
    printer.render_pdf("カレーの作り方", "2026年11月22日", "- 玉ねぎを炒める\n- 煮込む 20 分", path)
    data = path.read_bytes()
    assert data.startswith(b"%PDF")
    assert b"/MediaBox [0 0 595" in data  # A4（595 × 842 ポイント）
    if shutil.which("pdftotext"):  # 字が豆腐や白紙になっていない（2026-11-22 に字が点ほどになった）
        text = subprocess.run(
            ["pdftotext", str(path), "-"], capture_output=True, text=True, check=True
        ).stdout
        text = unicodedata.normalize(
            "NFKC", text
        )  # 字形が部首の字（⽅）として読み出されることがある
        assert "カレーの作り方" in text and "煮込む" in text


def test_a_bullet_mark_is_removed_only_once():
    html = printer.page_html("題", "日付", "- -3度まで下がる\n* *注意* 火を止める")
    assert "<li>-3度まで下がる</li>" in html and "<li>*注意* 火を止める</li>" in html


def test_the_body_lines_become_list_items():
    html = printer.page_html("題", "日付", "- ひとつめ\n- ふたつめ\nふつうの文")
    assert html.count("<li>") == 2 and "ふつうの文" in html
    assert "<h1>題</h1>" in html


def test_words_are_escaped_not_read_as_markup():
    html = printer.page_html("<b>題</b>", "日付", "1 < 2 & 3")
    assert "<b>題</b>" not in html and "&lt;b&gt;" in html and "1 &lt; 2 &amp; 3" in html


def test_the_job_goes_to_the_printer_in_black_and_white_one_sided(tmp_path):
    seen = {}

    def run(cmd, **kw):
        seen["cmd"] = cmd
        seen["env"] = kw.get("env") or {}
        return SimpleNamespace(
            returncode=0, stdout="request id is Brother_DCP_J552N-42 (1 file(s))\n", stderr=""
        )

    pdf = tmp_path / "out.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    job = printer.send_to_printer(pdf, printer="Brother_DCP_J552N", run=run)
    assert job == "Brother_DCP_J552N-42"
    cmd = seen["cmd"]
    assert cmd[:3] == ["lp", "-d", "Brother_DCP_J552N"] and cmd[-1] == str(pdf)
    opts = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-o"]
    assert {"print-color-mode=monochrome", "sides=one-sided", "media=A4"} <= set(opts)
    assert (
        seen["env"].get("LC_ALL") == "C"
    )  # 受付番号を英語で受ける（日本語では「リクエスト ID は …」）


def test_a_refused_job_raises_with_the_reason(tmp_path):
    def run(cmd, **kw):
        return SimpleNamespace(returncode=1, stdout="", stderr="lp: プリンターが見つかりません\n")

    with pytest.raises(printer.PrintError, match="見つかりません"):
        printer.send_to_printer(tmp_path / "x.pdf", printer="無い", run=run)


def test_a_hung_lp_is_an_error(tmp_path):
    def run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout", 0))

    with pytest.raises(printer.PrintError):
        printer.send_to_printer(tmp_path / "x.pdf", printer="p", run=run)


def test_without_a_qt_app_the_pdf_is_refused(tmp_path, monkeypatch):
    from PySide6.QtGui import QGuiApplication

    monkeypatch.setattr(QGuiApplication, "instance", staticmethod(lambda: None))
    with pytest.raises(printer.PrintError):
        printer.render_pdf("題", "日付", "本文", tmp_path / "x.pdf")
