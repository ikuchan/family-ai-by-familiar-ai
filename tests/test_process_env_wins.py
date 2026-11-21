"""環-an：起動のときに渡した DB の向き先（`DATABASE_URL`）は、`.env` と上書きファイル（`FAMILIAR_ENV_FILE`）より勝つ。

`.env` を `override=True` で読んでいたので、`DATABASE_URL=… FAMILIAR_ENV_FILE=.env.quiet ./run-gui.sh` としても `.env`
の値が勝ち、実機の確かめでテスト DB に向けたつもりが本番 DB を見ていた（2026-10-21 の申し送り）。
試験のプロセスは、テスト DB の向き先を渡されて始まる（`scripts/run_tests.sh`）。その値が、ファイルの本番らしい向き先に勝つことを見る。
"""

from __future__ import annotations

import logging
import os

import familiar_agent
from familiar_agent import bootstrap, env_reload

_PROD_LIKE = "postgresql://someone:sekret@prod-like-host:5432/familiar_ai"


def test_the_launch_value_is_the_test_db() -> None:
    assert familiar_agent.LAUNCH_DATABASE_URL == os.environ["DATABASE_URL"]


def test_the_launch_db_wins_over_the_env_file_and_warns_without_the_password(
    tmp_path, monkeypatch, caplog
) -> None:
    env = tmp_path / ".env"
    env.write_text(f"DATABASE_URL={_PROD_LIKE}\nENV_AN_OTHER=file\n", encoding="utf-8")
    monkeypatch.delenv("FAMILIAR_ENV_FILE", raising=False)
    monkeypatch.setenv("ENV_AN_OTHER", "given")
    given = os.environ["DATABASE_URL"]
    with caplog.at_level(logging.WARNING, logger="familiar_agent.bootstrap"):
        bootstrap.load_app_bootstrap(env)
    assert os.environ["DATABASE_URL"] == given
    assert os.environ["ENV_AN_OTHER"] == "file"  # DB の向き先のほかは、いままでどおりファイルが勝つ
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "prod-like-host" in text and "sekret" not in text


def test_the_launch_db_wins_over_the_overlay_file(tmp_path, monkeypatch) -> None:
    """報告された形：素の .env と上書きファイルの両方が DB の向き先を持つ。"""
    base = tmp_path / ".env"
    base.write_text(f"DATABASE_URL={_PROD_LIKE}\n", encoding="utf-8")
    overlay = tmp_path / ".env.quiet"
    overlay.write_text(f"DATABASE_URL={_PROD_LIKE}2\n", encoding="utf-8")
    monkeypatch.setattr(bootstrap, "_base_env_path", lambda: base)
    monkeypatch.setenv("FAMILIAR_ENV_FILE", str(overlay))
    given = os.environ["DATABASE_URL"]
    bootstrap.load_app_bootstrap(overlay)
    assert os.environ["DATABASE_URL"] == given


def test_the_launch_db_wins_over_a_reload(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text(f"DATABASE_URL={_PROD_LIKE}\n", encoding="utf-8")
    monkeypatch.delenv("FAMILIAR_ENV_FILE", raising=False)
    given = os.environ["DATABASE_URL"]
    env_reload.reload_env(env)
    assert os.environ["DATABASE_URL"] == given
