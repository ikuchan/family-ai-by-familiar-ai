"""拡散想起 (B) エンティティ辺の純ロジック（LLM フリー・機械的）。

現 W の MI の視点列（writer_id/subject_id/participants_json）から、再想起の種にする
person を選ぶ。話題の主体 subject を最優先、次いで participants、writer。自分・現話者・
既定話者は除外する（役割1 の r で既に効く話者を二重に数えない）。想起本体は後続。
"""

from __future__ import annotations


def select_entity_seeds(
    perspectives: "list[dict]", exclude: "set[str] | frozenset[str] | None" = None
) -> "list[str]":
    """視点列から再想起の種 person を選ぶ。subject 優先・participants・writer の順で重複除去。"""
    ex = set(exclude or ())
    seen: set[str] = set()
    out: list[str] = []

    def add(pid) -> None:
        p = str(pid) if pid else ""
        if p and p not in ex and p not in seen:
            seen.add(p)
            out.append(p)

    for row in perspectives:              # 1) subject（話題の主体）最優先
        add(row.get("subject_id"))
    for row in perspectives:              # 2) participants（在席者）
        for pid in (row.get("participants") or []):
            add(pid)
    for row in perspectives:              # 3) writer（書き手）
        add(row.get("writer_id"))
    return out
