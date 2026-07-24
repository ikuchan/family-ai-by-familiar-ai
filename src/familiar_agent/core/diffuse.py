"""拡散想起 (B) エンティティ辺の純ロジック（LLM フリー・機械的）。

現 W の MI の視点列（writer_id/subject_id/participants_json）から、再想起の種にする
person を選ぶ。話題の主体 subject を最優先、次いで participants、writer。自分・現話者・
既定話者は除外する（役割1 の r で既に効く話者を二重に数えない）。想起本体は後続。
"""

from __future__ import annotations

from typing import Callable


def diffuse_ids(
    seed_ids: "list[str]",
    get_candidates: "Callable[[list[str]], list[str]]",
    *,
    max_add: int,
    max_depth: int,
) -> "list[str]":
    """有界再帰の spreading：seed から候補を辿り、現在集合に無い MI を最大 max_add 件足す。

    `get_candidates(known)`＝現在の既知集合から候補 id を返すコールバック（(A)+(B) を合成）。
    足した id を含めて再度候補を取り（想起が想起を呼ぶ）、深さ max_depth・件数 max_add で打ち切る。
    追加した順の id リストを返す（seed 自身・重複は除く）。DB も採点も持たない純関数。
    """
    known = list(dict.fromkeys(str(s) for s in seed_ids if s))
    known_set = set(known)
    added: list[str] = []
    for _ in range(max(0, max_depth)):
        if len(added) >= max_add:
            break
        fresh = [
            c for c in dict.fromkeys(str(x) for x in get_candidates(known) if x)
            if c not in known_set
        ]
        if not fresh:
            break
        for c in fresh[: max_add - len(added)]:
            known.append(c)
            known_set.add(c)
            added.append(c)
    return added


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
