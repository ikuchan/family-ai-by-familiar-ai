"""#10 致命的エラー方針：DB 接続は2回リトライ、失敗なら明確な致命エラー。埋め込みは致命。"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from familiar_agent.errors import FatalStartupError


def test_connect_with_retry_succeeds_first_try():
    from familiar_agent import db

    with patch.object(db.psycopg2, "connect", return_value="CONN") as m:
        got = db._connect_with_retry("url", attempts=3, delay=0.0)
    assert got == "CONN"
    assert m.call_count == 1


def test_connect_with_retry_recovers_after_failures():
    from familiar_agent import db

    calls = {"n": 0}

    def _flaky(url):
        calls["n"] += 1
        if calls["n"] < 3:
            raise db.psycopg2.OperationalError("refused")
        return "CONN"

    with patch.object(db.psycopg2, "connect", side_effect=_flaky):
        got = db._connect_with_retry("url", attempts=3, delay=0.0)
    assert got == "CONN"
    assert calls["n"] == 3  # 2回リトライで3回目に成功


def test_connect_with_retry_raises_fatal_after_exhaustion():
    from familiar_agent import db

    with patch.object(db.psycopg2, "connect",
                      side_effect=db.psycopg2.OperationalError("refused")):
        with pytest.raises(FatalStartupError) as ei:
            db._connect_with_retry("postgresql://u:pw@h/db", attempts=3, delay=0.0)
    msg = str(ei.value)
    assert "PostgreSQL" in msg          # 何が起きたか
    assert "pw" not in msg              # パスワードは伏せる
