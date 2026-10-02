"""人物表の統合（知-af 段 4・2026-10-02・本人の決定ア）。`scripts/merge_persons.py` の中身。

本番の人物表は二重になっていた：記憶の付いた古い行と、いまの `FAMILY.md` が作った記憶 0 件の新しい行。**記憶の付いた
古い行を残し**、名前と呼び方を `FAMILY.md` に合わせ、記憶 0 件の行とテンプレートから作られたゴミの行を消す。記憶の
付け替えはしない（動かすのは人物表だけ）。

- **参照は決め打ちにしない**（`references`）：人物表を参照する外部キーを DB から読み、人物ごとに数える。スキーマが
  変わっても、参照のある行を消さない。参照が 1 件でもある行は消さない。
- **止まる場合**（`Plan.stops`）：1 人の家族に参照のある行が 2 つ以上ある（統合ではなく付け替えが要る）・参照のある行が
  2 人の家族に当たる。止まったら何も書かない。
- 計画（`make_plan`）は純関数。実行（`apply_plan`）は消す → 名前を書き換える の順（名前は一意なので）で、コミットは
  呼び手（スクリプトの `--apply`）がする。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import psycopg2.extras

from ..core.speaker_claim import aliases_of
from ..person_memory_manager import AGENT_SELF_ID, DEFAULT_PERSON_ID


@dataclass(frozen=True)
class Rename:
    id: str
    old_name: str
    new_name: str
    new_display: str


@dataclass(frozen=True)
class Delete:
    id: str
    name: str
    why: str


@dataclass
class Plan:
    renames: "list[Rename]" = field(default_factory=list)
    deletes: "list[Delete]" = field(default_factory=list)
    stops: "list[str]" = field(default_factory=list)  # 1 つでもあれば何も書かない
    notes: "list[str]" = field(default_factory=list)  # 残す行の知らせ


def _aliases(row: dict) -> "set[str]":
    out = {str(row.get("name") or "").strip()}
    for a in str(row.get("display_name") or "").replace(",", "、").split("、"):
        out.add(a.strip())
    return {a for a in out if a}


def _junk(row: dict) -> bool:
    """テンプレートから作られたゴミの行（名前が `- **呼び方**：…`・括弧書き）。"""
    name = str(row.get("name") or "").strip()
    return not name or name[0] in "-*（("


def make_plan(rows: "list[dict]", refs: "dict[str, int]", members: "list[dict]") -> Plan:
    """人物表の行・参照の数・`FAMILY.md` の家族から、統合の計画を立てる。"""
    plan = Plan()
    rows = [r for r in rows if str(r.get("id")) not in (AGENT_SELF_ID, DEFAULT_PERSON_ID)]
    groups: dict[int, list[dict]] = {i: [] for i in range(len(members))}
    for r in rows:
        n = int(refs.get(str(r["id"]), 0))
        if _junk(r):
            if n:
                plan.notes.append(f"「{r['name']}」はゴミの行に見えるが参照が {n} 件あるので残す")
            else:
                plan.deletes.append(
                    Delete(str(r["id"]), str(r["name"]), "テンプレートから作られた行")
                )
            continue
        hit = [i for i, m in enumerate(members) if _aliases(r) & set(aliases_of(m))]
        if len(hit) > 1:
            who = "・".join(str(members[i]["name"]) for i in hit)
            if n:
                plan.stops.append(f"「{r['name']}」（参照 {n} 件）が 2 人以上（{who}）に当たる")
            else:
                plan.deletes.append(
                    Delete(str(r["id"]), str(r["name"]), f"参照 0 件・{who} に当たる")
                )
            continue
        if not hit:
            plan.notes.append(
                f"「{r['name']}」は FAMILY.md の誰にも当たらない（参照 {n} 件）。残す"
            )
            continue
        groups[hit[0]].append(r)
    for i, m in enumerate(members):
        group = groups[i]
        name, display = str(m["name"]), str(m["display_name"])
        held = [r for r in group if refs.get(str(r["id"]), 0)]
        if len(held) > 1:
            plan.stops.append(
                f"「{name}」に参照のある行が 2 つ以上ある（"
                + "・".join(f"{r['name']} {refs[str(r['id'])]} 件" for r in held)
                + "）。統合ではなく付け替えが要る"
            )
            continue
        same = [r for r in group if r.get("name") == name]
        keep: "dict | None" = held[0] if held else (same[0] if same else None)
        for r in group:
            if r is keep:
                if r.get("name") != name or str(r.get("display_name") or "") != display:
                    plan.renames.append(Rename(str(r["id"]), str(r["name"]), name, display))
            else:
                plan.deletes.append(
                    Delete(str(r["id"]), str(r["name"]), f"参照 0 件・{name} に当たる")
                )
    return plan


def _foreign_keys(conn) -> "list[tuple[str, str]]":
    """人物表を参照する外部キー（表, 列）。DB から読む。"""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT c.conrelid::regclass::text AS tbl, a.attname AS col FROM pg_constraint c "
            "JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey) "
            "WHERE c.contype = 'f' AND c.confrelid = 'persons'::regclass ORDER BY 1, 2"
        )
        return [(str(r["tbl"]), str(r["col"])) for r in cur.fetchall()]


def referencing_tables(conn) -> "list[str]":
    """人物表を参照する表の名前。"""
    return sorted({tbl for tbl, _ in _foreign_keys(conn)})


def references(conn) -> "dict[str, int]":
    """人物ごとの参照の数（人物表を参照するすべての表の合計）。"""
    out: dict[str, int] = {}
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        for tbl, col in _foreign_keys(conn):
            cur.execute(f'SELECT "{col}" AS pid, count(*) AS n FROM {tbl} GROUP BY "{col}"')
            for r in cur.fetchall():
                out[str(r["pid"])] = out.get(str(r["pid"]), 0) + int(r["n"])
    return out


def persons(conn) -> "list[dict]":
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id, name, display_name, created_at FROM persons ORDER BY created_at")
        return [dict(r) for r in cur.fetchall()]


def apply_plan(conn, plan: Plan) -> None:
    """計画を当てる（消す → 名前を書き換える）。コミットはしない。止まる計画は当てない。"""
    if plan.stops:
        raise ValueError("止まる計画は当てない：" + "／".join(plan.stops))
    from . import clock

    now = clock.now_utc_iso()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        for d in plan.deletes:
            cur.execute("DELETE FROM persons WHERE id = %s", (d.id,))
        for r in plan.renames:
            cur.execute(
                "UPDATE persons SET name = %s, display_name = %s, updated_at = %s WHERE id = %s",
                (r.new_name, r.new_display, now, r.id),
            )
