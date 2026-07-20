"""PostgreSQL と sqlite3 の接続の差を吸収する薄いラッパ。

`ObservationMemory` は本番では RealDict カーソルの PostgreSQL 接続を、テストでは
sqlite3 接続を受け取る。呼び出し側が同じ書き方（`with conn.cursor() as cur:` と
辞書アクセス）で扱えるように、その差だけをここで吸収する。

SQL も想起ロジックも持たない。持たせると、ストアの実体（pgvector・BYTEA・次元）が
機構側へ漏れる。
"""

from __future__ import annotations

import psycopg2.extras


class _RealDictConnWrapper:
    """Wraps a psycopg2 connection so that cursor() always uses RealDictCursor."""

    def __init__(self, conn) -> None:
        self._conn = conn

    def cursor(self, **kwargs):
        kwargs.setdefault("cursor_factory", psycopg2.extras.RealDictCursor)
        return self._conn.cursor(**kwargs)

    def commit(self):   return self._conn.commit()
    def rollback(self): return self._conn.rollback()
    def close(self):    return self._conn.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


class _SQLiteCursorWrapper:
    """Wraps sqlite3.Cursor: adds context-manager support and %s→? translation."""

    def __init__(self, cur) -> None:
        self._cur = cur

    def execute(self, sql: str, params=None) -> None:
        sql = sql.replace("%s", "?")
        if params is None:
            self._cur.execute(sql)
        else:
            self._cur.execute(sql, params)

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self._cur.close()

    def __iter__(self):
        return iter(self._cur)


class _SQLiteConnWrapper:
    """Wraps a raw sqlite3.Connection for use in methods that expect psycopg2 style."""

    def __init__(self, conn) -> None:
        self._conn = conn

    def cursor(self, **kwargs) -> "_SQLiteCursorWrapper":
        return _SQLiteCursorWrapper(self._conn.cursor())

    def commit(self):   return self._conn.commit()
    def rollback(self): return self._conn.rollback()
    def close(self):    return self._conn.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass
