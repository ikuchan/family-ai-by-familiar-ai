"""試験 DB（5433）への接続の助け。試験ファイルごとに同じ形を書き写していた（23 本・2026-11-23 にまとめた）。"""

from __future__ import annotations

import os

import psycopg2
import psycopg2.extras


def autocommit_conn():
    """自動確定で、行を辞書で返す接続。行き先は conftest が決めた `DATABASE_URL`（試験 DB）。"""
    conn = psycopg2.connect(
        os.environ["DATABASE_URL"], cursor_factory=psycopg2.extras.RealDictCursor
    )
    conn.autocommit = True
    return conn
