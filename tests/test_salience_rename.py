"""概念5「顕著性」（salience・記号 s）への改名。

W を DB へ溜めていた旧方式のテーブルである。`source='working_memory'` で毎回消して
入れ直す形で、`memory.py` のコメントが「W は O からの派生ビューで毎ターン作り直すので、
このテーブルに溜める形自体が変わる」と撤去予定を明記している。

**撤去予定でも概念は残る**ので名前を与える。`activation` という語が5つの別の量に相乗り
していた状態を解くのが目的であり、消す予定のものだけ旧名のまま残すと、読む側は「これも
根づきの一種か」と取り違える。

ここでは (1) 表と列が新しい名前になっていること、(2) 旧名が残っていないこと、(3) 既存の
読み書き経路が新しい名前で動くこと、を見る。
"""

from __future__ import annotations

import os
import pathlib

import psycopg2


def _pg():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.autocommit = True
    return conn


def _tables() -> set[str]:
    conn = _pg()
    with conn.cursor() as cur:
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
        names = {r[0] for r in cur.fetchall()}
    conn.close()
    return names


def _columns(table: str) -> set[str]:
    conn = _pg()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
            (table,),
        )
        cols = {r[0] for r in cur.fetchall()}
    conn.close()
    return cols


def test_table_is_renamed() -> None:
    """表は `memory_salience`。旧名は残っていない。"""
    tables = _tables()
    assert "memory_salience" in tables, "memory_salience が無い（マイグレーション未適用）"
    assert "memory_activation" not in tables, "旧表 memory_activation が残っている"


def test_column_is_renamed() -> None:
    """列は `salience`。旧名は残っていない。"""
    cols = _columns("memory_salience")
    assert "salience" in cols, "salience 列が無い"
    assert "activation" not in cols, "旧列 activation が残っている"


def test_working_memory_read_uses_the_new_names() -> None:
    """W の読み出しが新しい表と列で通る。

    書き込みは `refresh_working_memory` が想起を伴うため、ここでは行を直接置いて
    読み出し側（`get_working_memory`）が新しい名前で引けることを見る。
    """
    import uuid

    from familiar_agent.tools.memory import ObservationMemory

    obs_id = str(uuid.uuid4())
    conn = _pg()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO observations (id, content, timestamp, direction, kind, emotion)"
            " VALUES (%s, %s, NOW(), %s, %s, %s)",
            (obs_id, "顕著性の読み出しテスト", "unknown", "observation", "neutral"),
        )
        cur.execute(
            "INSERT INTO memory_salience (id, memory_id, salience, source, context, activated_at)"
            " VALUES (%s, %s, %s, 'working_memory', %s, %s)",
            (str(uuid.uuid4()), obs_id, 0.7, "テスト", "2026-07-31T00:00:00+00:00"),
        )
    conn.close()

    got = ObservationMemory().get_working_memory()
    hit = [r for r in got if str(r.get("memory_id")) == obs_id]
    assert hit, "書いた W が読み出せない"
    assert "salience" in hit[0], "読み出しの列名が salience になっていない"


def test_old_names_are_gone_from_source() -> None:
    """旧名がソースとテストに残っていない（マイグレーションは凍結物なので対象外）。"""
    root = pathlib.Path(__file__).resolve().parents[1]
    stale = []
    for sub in ("src", "tests"):
        for path in (root / sub).rglob("*.py"):
            # 旧名を**検証の対象として**文字列で持つテストは除く。理由を1件ずつ挙げる。
            #   - このテスト自身：改名できたことを旧名で確かめる
            #   - 029 のテスト：凍結マイグレーションが旧名を持つ前提そのものを確かめる
            if path.name in (pathlib.Path(__file__).name,
                             "test_migration_029_utc_text_timestamps.py",
                             "test_six_concepts_vocabulary.py"):
                continue
            if "memory_activation" in path.read_text(encoding="utf-8"):
                stale.append(str(path.relative_to(root)))
    assert not stale, "旧名 memory_activation が残っている:\n" + "\n".join(sorted(stale))
