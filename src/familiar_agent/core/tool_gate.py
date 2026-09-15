"""話者ゲート——個人ティアの道具は、その人のターン以外では**存在しない**（知-f・2026-09-14）。

`設計方針_家の記録との接続` §3。MCP のサーバーからは誰が話しているか見えないので、ゲートは
こちらが掛ける。規則は名前で決まる：道具名が `_<誰かの英字>` で終わればその人の個人ティア
（`ask_vault_yusuke`）。英字は FAMILY.md の `- **英字**：Yusuke Ikunaga`（`parse_family_md` の
`latin`）。**description で頼むのではなく、定義の一覧から落とす**——存在しない道具は呼べない。
話者が分からなければ個人ティアは全部落とす（安全側）。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def gate_personal_tools(defs: list[dict], *, speaker: str, members: list[dict]) -> list[dict]:
    """個人ティアの道具を、その人が話しているときだけ残す。家族ティアはそのまま。"""
    # 本人の名は `名前` と `呼び方` のどちらでも通す。在席表が返すのは呼び方（パパ）で、
    # 名前（ゆうすけ）と比べると本人のターンでも落ちる（2026-09-15 実機）。
    owners = {
        m["latin"]: {str(m.get("name") or ""), str(m.get("display_name") or "")} - {""}
        for m in members
        if m.get("latin")
    }
    if not owners:
        return list(defs)
    out: list[dict] = []
    for d in defs:
        name = str(d.get("name", ""))
        owner = next((who for latin, who in owners.items() if name.endswith(f"_{latin}")), None)
        if owner is None or speaker in owner:
            out.append(d)
        else:
            logger.debug("話者ゲート：%s を落とした（話者=%s）", name, speaker or "不明")
    return out
