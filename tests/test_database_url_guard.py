"""テストの中で `DATABASE_URL` が変わったら、その場で落とす（環-r・2026-09-19）。

`reload_env()` が本物の `.env` を読み、`DATABASE_URL` が本番（5432）に変わったまま後のテストが走り、
本番 `familiar_ai` に 168 行の観測が書かれた（00:22〜01:21 JST・片付け済み）。`conftest` の番人は各テストの
後に `os.environ["DATABASE_URL"]` を確かめ、テスト DB の URL でなければ失敗させる（本番へ書く前に止める）。
"""

from __future__ import annotations

import os

import pytest

from tests import conftest


def test_the_guard_notices_a_changed_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://familiar:x@localhost:5432/familiar_ai")
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        conftest.assert_database_url_untouched()


def test_the_guard_is_quiet_when_the_url_is_the_test_one():
    assert os.environ["DATABASE_URL"] == conftest._TEST_DB_URL
    conftest.assert_database_url_untouched()
