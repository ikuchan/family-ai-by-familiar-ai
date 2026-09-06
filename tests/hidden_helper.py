"""テストから「現行でない」を作る・読むための素の SQL（段 2）。

`observations.superseded_by` 列は落ちた（`設計方針_MI間の関係` v0.3）。畳む印は関係
（`relations` と `relation_members`）にあり、**役割 `旧` が「もう現行ではない」を表す**。

層を通さず素の SQL で書くのは、層の振る舞いを検査する側が層に依存すると、層が壊れた
ときにテストも一緒に壊れて何も分からなくなるためである。
"""

from __future__ import annotations


def hide(cur, old_id: str, new_id: str, kind: str = "未分類") -> bool:
    """`old_id` を現行から外し、`new_id` を新しい側にする。外したなら True。

    **先着が勝つ。** 本番の `RelationStore.hide` と同じにしておく。固定の id を使う検査は
    テスト DB に前回の関係を残すので、ここで落ちると二度目の実行から赤くなる。
    """
    if is_hidden(cur, old_id):
        return False
    cur.execute(
        "WITH r AS (INSERT INTO relations (kind) VALUES (%s) RETURNING id) "
        "INSERT INTO relation_members (relation_id, obs_id, role, position) "
        "SELECT r.id, v.obs, v.role, v.pos FROM r, "
        "(VALUES (%s, '旧', 0), (%s, '新', 1)) AS v(obs, role, pos)",
        (kind, old_id, new_id),
    )
    return True


def hidden_by(cur, old_id: str) -> "str | None":
    """`old_id` を外した相手（新しい側）を返す。外れていなければ None。"""
    cur.execute(
        "SELECT m2.obs_id AS new_id FROM relation_members m1 "
        "JOIN relation_members m2 "
        "  ON m2.relation_id = m1.relation_id AND m2.role = '新' "
        "WHERE m1.obs_id = %s AND m1.role = '旧'",
        (old_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return str(row["new_id"] if hasattr(row, "keys") else row[0])


def is_hidden(cur, old_id: str) -> bool:
    """その観測がもう現行でないか。"""
    cur.execute("SELECT 1 FROM relation_members WHERE obs_id = %s AND role = '旧'", (old_id,))
    return cur.fetchone() is not None


# 想起の絞りの、テスト側での写し。`store.relations.not_hidden` と同じ形。
LIVE = (
    "NOT EXISTS (SELECT 1 FROM relation_members _rm "
    "WHERE _rm.obs_id = {alias}.id AND _rm.role = '旧')"
)


def has_superseded_by_column() -> bool:
    """`observations.superseded_by` がまだ有るか。

    059 が落としたので、通常は偽である。旧マイグレーションを**再実行**して確かめる
    検査は、書き込み先の列が無くなった時点で対象を失う。列を戻した人が居れば、
    その検査はまた動く。
    """
    import os

    import psycopg2

    with psycopg2.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'observations' AND column_name = 'superseded_by'"
        )
        return cur.fetchone() is not None
