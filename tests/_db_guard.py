"""テストの接続の番人（環-ae・2026-10-08）。テスト DB（5433）以外へは、接続する前に拒む。

環境変数を空にしたテストで接続先が `db.py` の既定（本番 5432）に落ち、本番の設定値を読んだ（`test_arbiter_timeout` を
単独で回したとき）。`assert_database_url_untouched`（環-r）は環境変数しか見ないので見えなかった。ここでは環境変数では
なく**接続先そのもの**を見る。拒んだことは記録し、呼び手が例外を飲み込んでも `conftest` の後ろの番人がそのテストを落とす。
"""

from __future__ import annotations

from typing import Any

import psycopg2

#: テスト DB のある口。URL はこの前置きで始まるものだけ通す（ワーカーごとの DB 名も含む）。
TEST_HOST_PREFIX = "postgresql://familiar:familiar@localhost:5433/"
_TEST_PORT = "5433"
_LOCAL_HOSTS = ("localhost", "127.0.0.1")

_real_connect = psycopg2.connect
_blocked: list[str] = []


class ProductionConnectionBlocked(RuntimeError):
    """テスト DB 以外への接続を拒んだ。"""


def _allowed(dsn: Any, kwargs: dict) -> bool:
    if isinstance(dsn, str) and dsn:
        return dsn.startswith(TEST_HOST_PREFIX)
    if dsn is None:
        return str(kwargs.get("port")) == _TEST_PORT and kwargs.get("host") in _LOCAL_HOSTS
    return False


def _mask(dsn: Any, kwargs: dict) -> str:
    if isinstance(dsn, str):
        head, sep, tail = dsn.rpartition("@")
        return f"…@{tail}" if sep else dsn
    return f"host={kwargs.get('host')} port={kwargs.get('port')} dbname={kwargs.get('dbname')}"


def _guarded_connect(dsn: Any = None, *args: Any, **kwargs: Any) -> Any:
    if dsn is None and "dsn" in kwargs:
        dsn = kwargs.pop("dsn")
    if not _allowed(dsn, kwargs):
        where = _mask(dsn, kwargs)
        _blocked.append(where)
        raise ProductionConnectionBlocked(
            f"テスト DB 以外へつなごうとした（{where}）。接続する前に止めた"
        )
    return _real_connect(dsn, *args, **kwargs)


_guarded_connect.familiar_guard = True  # type: ignore[attr-defined]


def install() -> None:
    """`psycopg2.connect` を番人に替える（`conftest` が import のときに 1 回呼ぶ）。"""
    psycopg2.connect = _guarded_connect  # type: ignore[assignment]


def forget_blocked() -> None:
    _blocked.clear()


def assert_no_blocked_connection() -> None:
    """このテストで拒んだ接続があれば落とす（記録は消す）。"""
    if not _blocked:
        return
    seen = "・".join(_blocked)
    _blocked.clear()
    raise RuntimeError(
        f"テスト DB 以外へつなごうとした（{seen}）。環境変数を空にするなら DATABASE_URL を残す"
        "（`tests/_db_guard` の番人が接続する前に止めた）"
    )
