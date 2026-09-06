"""`observations.superseded_by` を関係へ移し、列を落とす（`設計方針_MI間の関係` v0.3・段 2）。

**一つの列に四つの意味が乗っていた。** 版チェーンの改訂、答えの逐語を会話要約へ入れる
畳み込み、配れた保留発話の解決、鎖の前進である。畳まれた側が誤りになったのは改訂だけで、
残りは畳まれた時点の事実として正しい。列のままでは、この区別を書き分ける場所が無い。

関係へ移すと、種類が理由を言い、**役割 `旧` が「もう現行ではない」を表す**。想起の絞りは
種類を見ずに役割だけを見るので、結合の要らない一文になる。

**既存の辺は `未分類` で移す。** どの書き手が作ったかは行から判別できない。置換先の
`direction` から推し量ることはできるが、推測を事実として書き込むと、あとから見て区別が
つかなくなる。意味の分かるものは段 3 以降で付け替える。

**列を落とす前に控えを取る。** 本番の列を落とす操作は戻せない。移し損ねがあっても
`observations_superseded_by_backup` から辿れるようにしておく
（`observations_removed_fillers`（054）と同じやり方）。移した件数と控えた件数が合わなければ
例外で止め、列を落とさない。

部分一意索引 `idx_relation_members_old` は、いま列の `WHERE superseded_by IS NULL` が
持っていた**先着が勝つ**（既に解決済みの行を張り替えない）を引き継ぐ。役割の語は種類を
またいで重ならないので、`旧` を表の全体で一意にして差し支えない。
"""

from __future__ import annotations

_BACKUP = "observations_superseded_by_backup"


def _cols(row) -> list:
    """1行を列の順に読む。

    マイグレーションを流す接続は、経路によって tuple を返したり dict を返したりする
    （アプリ側は `RealDictCursor` で包み、テストの土台は素の接続で流す）。どちらでも
    同じに読めるようにしておかないと、行が1件でもあった瞬間に落ちる。
    """
    return list(row.values()) if hasattr(row, "values") else list(row)


def edges(conn) -> "list[tuple[str, str]]":
    """列が持っている (旧, 新) の組を読む。"""
    with conn.cursor() as cur:
        cur.execute("SELECT id, superseded_by FROM observations WHERE superseded_by IS NOT NULL")
        return [(str(c[0]), str(c[1])) for c in (_cols(r) for r in cur.fetchall())]


def move_edges(conn, pairs: "list[tuple[str, str]]") -> int:
    """(旧, 新) の組を `未分類` の関係として書く。書いた本数を返す。

    ヘッダと項を1文で書く。分けると、途中で落ちたときに項の無いヘッダが残る。
    """
    n = 0
    with conn.cursor() as cur:
        for old, new in pairs:
            cur.execute(
                "WITH r AS (INSERT INTO relations (kind) VALUES ('未分類') RETURNING id) "
                "INSERT INTO relation_members (relation_id, obs_id, role, position) "
                "SELECT r.id, v.obs, v.role, v.pos FROM r, "
                "(VALUES (%s, '旧', 0), (%s, '新', 1)) AS v(obs, role, pos)",
                (old, new),
            )
            n += 1
    return n


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {_BACKUP} AS "
            "SELECT id, superseded_by FROM observations WHERE superseded_by IS NOT NULL"
        )
        cur.execute(f"SELECT count(*) AS n FROM {_BACKUP}")
        backed = int(_cols(cur.fetchone())[0])

    pairs = edges(conn)
    moved = move_edges(conn, pairs)
    if moved != backed:
        raise RuntimeError(f"059: 控えた {backed} 本に対して移せたのは {moved} 本。列を落とさない")

    with conn.cursor() as cur:
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_relation_members_old "
            "ON relation_members(obs_id) WHERE role = '旧'"
        )
        cur.execute("ALTER TABLE observations DROP COLUMN IF EXISTS superseded_by")
