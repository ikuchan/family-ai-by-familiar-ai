"""テストはテスト DB（5433）以外へつながない（環-ae・2026-10-08）。

環境変数を空にしたテスト（`patch.dict(os.environ, {}, clear=True)`）の中で `Config` を作ると、層 3 の設定値を DB へ
読みに行く。`DATABASE_URL` が無いので接続先が `db.py` の既定（本番 5432）に落ち、`test_arbiter_timeout` を単独で回すと
本番の 4.5（その晩の REST が書いた値）を読んで落ちた。番人（`assert_database_url_untouched`）は環境変数しか見ず、
`patch.dict` は抜けるときに戻すので見えなかった。番人を接続の口（`psycopg2.connect`）に置き、テスト DB 以外は
**接続する前に**拒む。呼び手が例外を飲んでも、止めた記録でそのテストを落とす。

**本物の本番へはつながない**：最初に番人が入っていることを確かめ、入っていなければそこで落とす。
"""

from __future__ import annotations

import os

import psycopg2
import pytest

from familiar_agent import db
from tests import _db_guard

PROD = "postgresql://familiar:familiar@localhost:5432/familiar_ai"


@pytest.fixture
def real(monkeypatch):
    """番人の先（本物の接続）を記録に差し替える。テスト DB に通す場合も、本当にはつながない。"""
    assert getattr(psycopg2.connect, "familiar_guard", False), "番人が入っていない"
    calls: list = []
    monkeypatch.setattr(_db_guard, "_real_connect", lambda *a, **kw: calls.append((a, kw)))
    yield calls
    _db_guard.forget_blocked()


def test_the_guard_is_on_the_connect_itself():
    assert getattr(psycopg2.connect, "familiar_guard", False)


def test_a_production_url_is_refused_before_connecting(real):
    with pytest.raises(_db_guard.ProductionConnectionBlocked):
        psycopg2.connect(PROD)
    assert real == []


def test_it_holds_through_the_db_module_even_with_retries(real):
    """`db._connect_with_retry` は例外を受けて試し直すが、どの回も本番へは行かない。"""
    with pytest.raises(Exception):
        db._connect_with_retry(PROD, attempts=3, delay=0.0)
    assert real == []


def test_a_swallowed_refusal_still_fails_the_test(real):
    """呼び手が例外を飲み込んでも（`load_overrides` のように）、後ろの番人が名前つきで落とす。"""
    try:
        psycopg2.connect(PROD)
    except Exception:  # noqa: BLE001
        pass
    with pytest.raises(RuntimeError, match="5432"):
        _db_guard.assert_no_blocked_connection()
    _db_guard.assert_no_blocked_connection()  # 一度落としたら記録は消える


def test_the_test_db_goes_through(real):
    psycopg2.connect(os.environ["DATABASE_URL"])
    psycopg2.connect(host="localhost", port=5433, dbname="familiar_test")
    assert len(real) == 2


def test_keyword_style_production_is_refused_too(real):
    with pytest.raises(_db_guard.ProductionConnectionBlocked):
        psycopg2.connect(host="localhost", port=5432, dbname="familiar_ai")
    assert real == []


# ── 段 2（ア）：`db.py` に本番の既定を置かない ─────────────────────────────────────────


def test_without_database_url_the_db_refuses_instead_of_going_to_production(real):
    # 環境変数はこのテストの中で戻す（monkeypatch は後ろの番人より後に戻るので、環-r の番人に引っかかる）
    saved = os.environ.pop("DATABASE_URL")
    try:
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            db.Database().conn()
    finally:
        os.environ["DATABASE_URL"] = saved
    assert real == [] and _db_guard._blocked == []  # 接続を試してもいない


def test_no_production_url_is_written_in_the_source():
    import inspect

    assert "5432/familiar_ai" not in inspect.getsource(db)
