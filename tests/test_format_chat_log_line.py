"""Tests for format_chat_log_line — chat.log ファイル書き込み用タイムスタンプ付与。

Red: 修正前は format_chat_log_line が未定義 → ImportError で FAIL。
Green: 関数追加後 → PASS。
"""
from __future__ import annotations

import re
from datetime import datetime

from familiar_agent._ui_helpers import format_chat_log_line


def test_prepends_timestamp_in_applog_format():
    """app.log と同形式 [YYYY-MM-DD HH:MM:SS] を前置する。"""
    now = datetime(2026, 6, 15, 8, 43, 18)
    line = format_chat_log_line("[パジュ] おはよう", now=now)
    assert line == "[2026-06-15 08:43:18] [パジュ] おはよう"


def test_preserves_original_text():
    """元テキスト（プレフィックス含む）は時刻の後にそのまま残る。"""
    now = datetime(2026, 6, 15, 0, 0, 0)
    line = format_chat_log_line("👀 見てる...", now=now)
    assert line.endswith("👀 見てる...")
    assert line.startswith("[2026-06-15 00:00:00] ")


def test_default_now_is_used_when_omitted():
    """now 省略時は現在時刻が入る（形式だけ検証）。"""
    line = format_chat_log_line("x")
    assert re.match(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] x$", line)
