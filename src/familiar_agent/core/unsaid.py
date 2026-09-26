"""言いたかったのに言えなかったこと（出-as 段 7・2026-09-26・`設計方針_話していいかの決まり` §2.6）。純関数。

話そうとして止められた発話（誰も映っていない情動・窓が切れた後の返事）は、保留して人が来たら配るのをやめ、独り言
として記録する。想起で上がってくるよう、記録の時点で工夫する（本人の決定）：

- 中身を「**〇〇に言いたかったこと：**…」にする（関連 r）。相手が分からなければ「誰か」。
- 相手の面に立てる（人 p）——相手が分からなければパジュ自身の面。書くのはループ。
- 根づきを **2** にする（g・「大事」と 2 回申告された記録と同じ）。

返事を声に出した求めで、その記録を「使った」と申告したら（`referred`・`important`）、**畳んで普通の重さに戻す**。
同じことが何度も上がってこないように。
"""

from __future__ import annotations

#: 言いたかったことの印。畳んだ記録は頭に「伝えた：」が付くので、印があっても数えない。
MARK = "に言いたかったこと："
TOLD = "伝えた："
#: 記録するときの根づき（本人の決定 2026-09-26）。
GROUNDEDNESS = 2


def content(who: str, text: str) -> str:
    """記録の中身。相手が分からなければ「誰か」。"""
    return f"{who or '誰か'}{MARK}{text}"


def is_unsaid(text: str) -> bool:
    """言いたかったことの記録か（想起の頭「[そばに居た] 」などが付いていても見る）。畳んだものは外す。"""
    text = str(text or "")
    return MARK in text and TOLD not in text


def told(raw, w_id_map: "dict[str, str]", memories: list) -> "list[tuple[str, str]]":
    """申告のうち、言いたかったことを「使った」と言ったもの（記録の id と中身）。

    申告の id は W の 12 桁で、`w_id_map` で記録の id へ引く（`workspace.apply_memory_verdicts` と同じ）。
    """
    if not raw or not w_id_map:
        return []
    by_id = {
        str(getattr(r.mi, "obs_id", "")): str(getattr(r.mi, "content", "")) for r in memories or []
    }
    out: list[tuple[str, str]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        if str(item.get("verdict", "")).strip().lower() not in ("referred", "important"):
            continue
        full = w_id_map.get(str(item.get("id", "")).replace("-", "")[:12])
        if full and is_unsaid(by_id.get(full, "")) and all(full != f for f, _ in out):
            out.append((full, by_id[full]))
    return out
