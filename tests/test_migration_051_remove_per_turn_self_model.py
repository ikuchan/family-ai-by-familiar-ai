"""051：毎ターンの `self_model` をやめ、溜まった行を退避表へ移す。

**畳まれずに溜まり続けていた。** 2026-08-21 のダンプの `observations_removed_self_model`
には 1068 行（2026-06-08 〜 08-10）が入っており、`superseded_by` が付いていたのは
**1068 中 1 件だけ**である。中身は英語の短い自己記述で、`Nothing.` が 6 件あった——
`insight.lower() != "nothing"` の判定を、ピリオド付きの `Nothing.` がすり抜けていた。

```
Nothing.                                          6
I am polite.                                      2
I am concerned about your well-being.             2
I can see and move my head to look around.        2
```

**読み手はいなかった。** `recall_self_model` と `format_self_model_for_context` は本番
コードから呼ばれていない（2026-09-03 に確認）。毎ターン軽量LLM を呼んで書いていたが、
書いたものを誰も読んでいなかった。

**削除でなく退避である。** 表は `observations` と同じ列を持つ素の写し（制約も索引も無い）で、
戻せる形になっている。自己理解は capability manifest と REST 内省（記-a）が担う。
"""

from __future__ import annotations

import importlib.util
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]
_MIGRATION = "2026-08-15-051_remove_the_per_turn_self_model.py"


def _conn():
    c = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    c.autocommit = True
    return c


def _run_migration() -> None:
    path = Path(__file__).parent.parent / "migration" / _MIGRATION
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    conn = psycopg2.connect(_DB_URL)
    try:
        mod.upgrade(conn)
        conn.commit()
    finally:
        conn.close()


def _plant(direction: str, kind: str) -> str:
    obs_id = str(uuid.uuid4())
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction, kind, emotion) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (obs_id, f"051 {obs_id}", datetime.now(timezone.utc), direction, kind, "happy"),
            )
    finally:
        conn.close()
    return obs_id


def _exists(table: str, obs_id: str) -> bool:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT 1 FROM {table} WHERE id = %s", (obs_id,))
            return cur.fetchone() is not None
    finally:
        conn.close()


# ── ① 退避表がある ──────────────────────────────────────────────────────────

def test_the_quarantine_table_exists() -> None:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT to_regclass('observations_removed_self_model') AS t"
            )
            assert cur.fetchone()["t"] is not None
    finally:
        conn.close()


# ── ② 退避する（削除しない） ────────────────────────────────────────────────

def test_the_migration_moves_the_rows_instead_of_deleting_them() -> None:
    """内省の自己記述は退避表へ移り、`observations` から消える。戻せる形で残る。"""
    mine = _plant("内省", "self_model")
    other = _plant("会話", "conversation")

    _run_migration()

    assert not _exists("observations", mine), "観測から消えていない"
    assert _exists("observations_removed_self_model", mine), "退避表に残っていない"
    assert _exists("observations", other), "関係のない記録まで動かしている"


def test_the_migration_is_idempotent() -> None:
    """二度流しても壊れない（移すものが無いだけ）。"""
    _run_migration()
    _run_migration()


# ── ③④ 毎ターン書くのをやめた ──────────────────────────────────────────────

def test_the_agent_no_longer_writes_a_self_model_every_turn() -> None:
    """`_update_self_model` は無い。毎ターン軽量LLM を呼んで書く経路そのものを外した。"""
    from familiar_agent.agent import EmbodiedAgent

    assert not hasattr(EmbodiedAgent, "_update_self_model"), (
        "毎ターンの自己記述を書く経路が残っている"
    )


def test_the_self_model_prompt_is_gone() -> None:
    """使う人が居なくなったプロンプトも残さない（反証側）。"""
    import familiar_agent.agent as agent_module

    assert not hasattr(agent_module, "_SELF_MODEL_PROMPT")
