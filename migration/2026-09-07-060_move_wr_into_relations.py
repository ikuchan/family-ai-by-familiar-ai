"""共起（WR）を関係の器へ移す（`設計方針_MI間の関係` v0.7・段 5）。

記録どうしのつながりを表す独立した表を無くし、正本を一つにする。`wr_records` と
`wr_record_items`（030）は、既にヘッダと項に分ける形を取っていた。種類 `共起`・役割 `項`
の関係へ、形を変えずに移せる。

**位置は空ける。** 共起（ある反復で一緒に活性した記録の集まり）に前後は無い。

`wr_record_items.mi_id` は名前に反して**観測の id** である（拡散想起が
`o.id::text = i.mi_id` で結合していた）。`relation_members.obs_id` と同じ粒度なので、
粒度を揃えたまま移せる。

**元の2表は落とさず改名して残す。** 本番の表を落とす操作は戻せない。移し損ねがあっても
`wr_records_moved` と `wr_record_items_moved` から辿れるようにしておく
（`observations_removed_fillers`（054）と同じやり方）。移した項の数と元の項の数が合わ
なければ例外で止め、改名しない。
"""

from __future__ import annotations


def _cols(row) -> list:
    """1行を列の順に読む（接続によって tuple にも dict にもなる）。"""
    return list(row.values()) if hasattr(row, "values") else list(row)


def records(conn) -> "list[list[str]]":
    """WR ごとの項の並びを読む。"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT r.id, array_agg(i.mi_id) AS ids FROM wr_records r "
            "LEFT JOIN wr_record_items i ON i.wr_id = r.id GROUP BY r.id ORDER BY r.id"
        )
        out = []
        for row in cur.fetchall():
            ids = _cols(row)[1] or []
            out.append([str(m) for m in ids if m])
        return out


def move_records(conn, groups: "list[list[str]]") -> int:
    """項の並びを `共起` の関係として書く。書いた本数を返す。

    ヘッダと項を1文で書く。分けると、途中で落ちたときに項の無いヘッダが残る。
    """
    n = 0
    with conn.cursor() as cur:
        for ids in groups:
            uniq = list(dict.fromkeys(i for i in ids if i))
            if not uniq:
                continue
            cur.execute(
                "WITH r AS (INSERT INTO relations (kind) VALUES ('共起') RETURNING id) "
                "INSERT INTO relation_members (relation_id, obs_id, role, position) "
                "SELECT r.id, v.obs, '項', NULL FROM r, unnest(%s::text[]) AS v(obs)",
                (uniq,),
            )
            n += 1
    return n


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.wr_record_items') IS NOT NULL AS present")
        if not _cols(cur.fetchone())[0]:
            return  # 既に移してある
        cur.execute("SELECT count(*) AS n FROM wr_record_items")
        before = int(_cols(cur.fetchone())[0])

    groups = records(conn)
    move_records(conn, groups)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) AS n FROM relation_members m "
            "JOIN relations r ON r.id = m.relation_id AND r.kind = '共起'"
        )
        after = int(_cols(cur.fetchone())[0])
    moved = sum(len(dict.fromkeys(g)) for g in groups)
    if after < moved:
        raise RuntimeError(f"060: 移せたのは {after} 件で、移すはずの {moved} 件に足りない")
    if moved > before:
        raise RuntimeError(f"060: 元の {before} 件より多い {moved} 件を移している")

    with conn.cursor() as cur:
        cur.execute("ALTER TABLE wr_record_items RENAME TO wr_record_items_moved")
        cur.execute("ALTER TABLE wr_records RENAME TO wr_records_moved")
