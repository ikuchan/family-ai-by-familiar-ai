"""マイグレーションの自動適用を起動時に留める（`HOLD_MIGRATIONS`）。

`Database` は接続のたびに `apply_migrations` を無条件で呼ぶ。そのため `migration/` へ
ファイルを置いた時点で、次に本番へ繋いだプロセスが適用してしまい、適用の可否を人が
握れなかった（実際、列改名のマイグレーションが意図せず本番へ入った）。

留めたときは**警告を出して起動を続ける**。未適用のまま動くと、コードが新しいスキーマを
前提にしている場合に実行時へ跳ねるので、何が保留されているかを1件ずつ残す。件数だけでは
どれが保留か分からない。

ここでは (1) 未適用の列挙、(2) 留めたときに適用されないこと、(3) 保留の一覧が warning に
出ること、(4) 留めていても未適用が0件なら鳴らないこと、(5) 環境変数が無ければ従来どおり
適用されること、を見る。
"""

from __future__ import annotations

import logging
import os

import psycopg2
import pytest

from familiar_agent.db_migrations import (
    apply_migrations,
    pending_migration_ids,
)

_DB_URL = os.environ["DATABASE_URL"]


@pytest.fixture()
def conn():
    c = psycopg2.connect(_DB_URL)
    c.autocommit = True
    yield c
    c.close()


def _write_migration(tmp_path, name: str, body: str = "    pass\n"):
    """一意な id で書く。

    固定の id を使うと、過去の実行で `schema_migrations` に残った記録と衝突し、
    「未適用のはずが適用済み」に見える。DB はテスト間で残るので、id を毎回変える。
    """
    unique = f"{name}_{abs(hash(str(tmp_path))) % 10**8}"
    path = tmp_path / f"{unique}.py"
    path.write_text(f"def upgrade(conn) -> None:\n{body}", encoding="utf-8")
    return unique


def test_pending_lists_unapplied_in_order(conn, tmp_path) -> None:
    """未適用の id を辞書順で返す。`_` 始まりは対象外。"""
    beta = _write_migration(tmp_path, "2099-01-01-900_beta")
    gamma = _write_migration(tmp_path, "2099-01-01-901_gamma")
    (tmp_path / "_helper.py").write_text("def upgrade(conn) -> None:\n    pass\n",
                                         encoding="utf-8")

    assert pending_migration_ids(conn, tmp_path) == sorted([beta, gamma])


def test_pending_excludes_already_applied(conn, tmp_path) -> None:
    """適用済みは未適用に出ない。"""
    _write_migration(tmp_path, "2099-01-01-902_delta")
    apply_migrations(conn, tmp_path)
    assert pending_migration_ids(conn, tmp_path) == []


def test_hold_skips_apply_and_warns(conn, tmp_path, monkeypatch, caplog) -> None:
    """留めると適用されず、保留の id が1件ずつ warning に出る。"""
    from familiar_agent import db as db_mod

    mid = _write_migration(tmp_path, "2099-01-01-903_epsilon")
    monkeypatch.setenv("HOLD_MIGRATIONS", "1")

    with caplog.at_level(logging.WARNING, logger="familiar_agent.db"):
        applied = db_mod.apply_or_hold_migrations(conn, tmp_path)

    assert applied == 0, "留めたのに適用された"
    assert pending_migration_ids(conn, tmp_path) == [mid]
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert mid in text, "保留の id が警告に出ていない"


def test_hold_is_quiet_when_nothing_is_pending(conn, tmp_path, monkeypatch, caplog) -> None:
    """留めていても未適用が0件なら鳴らない（毎回鳴ると読まれなくなる）。"""
    from familiar_agent import db as db_mod

    monkeypatch.setenv("HOLD_MIGRATIONS", "1")
    with caplog.at_level(logging.WARNING, logger="familiar_agent.db"):
        applied = db_mod.apply_or_hold_migrations(conn, tmp_path)

    assert applied == 0
    assert not [r for r in caplog.records if "保留" in r.getMessage()], "空でも鳴っている"


def test_without_the_env_var_migrations_apply(conn, tmp_path, monkeypatch) -> None:
    """環境変数が無ければ従来どおり適用される（回帰の見張り）。"""
    from familiar_agent import db as db_mod

    monkeypatch.delenv("HOLD_MIGRATIONS", raising=False)
    _write_migration(tmp_path, "2099-01-01-904_zeta")

    assert db_mod.apply_or_hold_migrations(conn, tmp_path) == 1
    assert pending_migration_ids(conn, tmp_path) == []
