"""広げる道具のうち、機械が O から引くだけのもの（2026-11-23 に `event_loop.py` から移した）。

記憶の木の節の要約を読む（`recall_tree`・記-m 段 2）と、記録の前後の話を継起でたどる（`recall_thread`・記-q）。LLM は
使わない。どれも `agent` を受け取るだけで状態を持たず、呼ぶのはループ（`event_loop.InformationProcessing._run_widen`）。
"""

from __future__ import annotations

import logging

from .actions import THREAD_MAX_STEPS

logger = logging.getLogger(__name__)


def recall_tree(agent, tool_input: dict) -> str:
    """記憶の木の節を指して、その節の要約を読む（記-m 段 2）。機械が O から引くだけで、LLM は使わない。

    節ごとに見出し（【2025-08】）を付けて返す。書かれていない節は「まだ要約が無い」と返し、主LLM は
    日を指し直せる。人を指せばその人の面から（家族に無い名前は断る）、省けば家族全体。
    """
    nodes = tool_input.get("nodes") or []
    if isinstance(nodes, str):
        nodes = [nodes]
    name = str(tool_input.get("person") or "").strip()
    pid = None
    if name:
        pid = agent._pmm.find_person_id_by_name(name)
        if pid is None:
            return f"「{name}」は家族に見つからないので、その人の木はたどれない"
    whose = f"（{name}との要約）" if name else "（家族全体の要約）"
    parts: list[str] = []
    for node in [str(n).strip() for n in nodes if str(n).strip()]:
        rows = agent._oif.tree_summaries(node, person_id=pid)
        if rows is None:
            parts.append(f"【{node}】読めない（年 2025・月 2025-08・日 2025-08-15 の形で指す）")
        elif not rows:
            parts.append(f"【{node}】まだ要約が無い")
        else:
            parts.append(f"【{node}】\n" + "\n".join(f"- {r.get('content', '')}" for r in rows))
    if not parts:
        return "節が指されていない（年 2025・月 2025-08・日 2025-08-15 の形で指す）"
    return whose + "\n" + "\n".join(parts)


def recall_thread(agent, tool_input: dict) -> str:
    """記録の前後の話を継起でたどる（記-q・2026-10-15）。機械が O から引くだけで、LLM は使わない。

    件数は主LLM が決め、前後それぞれ `THREAD_MAX_STEPS`（10 件〔仮〕）で止める。止めたら返りの先頭にそう書く。
    """
    oif = agent._oif
    short = str(tool_input.get("id") or "").strip()
    full = oif.id_from_short(short) if short else None
    if full is None:
        return f"id「{short}」の記録は見つからない（作業状態に載った 12 桁の id を渡す）"

    def _steps(key: str) -> int:
        try:
            return max(0, int(tool_input.get(key) or 0))
        except (TypeError, ValueError):
            return 0

    before, after = _steps("before"), _steps("after")
    notes = []
    if before > THREAD_MAX_STEPS:
        notes.append(f"前は {THREAD_MAX_STEPS} 件までにした")
    if after > THREAD_MAX_STEPS:
        notes.append(f"後は {THREAD_MAX_STEPS} 件までにした")
    before, after = min(before, THREAD_MAX_STEPS), min(after, THREAD_MAX_STEPS)
    rows = oif.thread(full, before=before, after=after)
    logger.info("event-loop recall_thread：前 %d・後 %d 件", before, after)
    head = f"（{THREAD_MAX_STEPS} 件まで：" + "・".join(notes) + "）\n" if notes else ""
    if len(rows) <= 1:
        return head + "前後につながる記録が無い"

    def _label(off: int) -> str:
        return "【これ】" if off == 0 else (f"【前 {-off}】" if off < 0 else f"【後 {off}】")

    n_before = sum(1 for r in rows if int(r.get("offset", 0)) < 0)
    n_after = sum(1 for r in rows if int(r.get("offset", 0)) > 0)
    lines = [f"{_label(int(r.get('offset', 0)))}{r.get('content', '')}" for r in rows]
    return head + f"（id {short} の前 {n_before} 件・後 {n_after} 件）\n" + "\n".join(lines)
