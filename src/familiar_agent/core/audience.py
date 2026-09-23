"""宛先の条件——それを言うのに誰が要るか（出-ap・2026-09-23）。純関数。

実機 15:49、話者が `unconfirmed` のまま家の予定（フーコック旅行・キャンプのキャンセル）を
全部話した。**同じ発話の中で「どなたでしょうか？」と聞いている。** 在席の注記は誰かを
**呼ぶ**ことだけを止めており、**何を話してよいか**は誰も見ていなかった。

話したいことは、それを言うのに**誰が要るか**で 4 段に分かれる（本人・2026-09-23）。

| 段 | 条件 | 例 |
|---|---|---|
| 0 | 誰もいなくても | 独り言 |
| 1 | 誰かいたら | タイマーが鳴っている |
| 2 | 家族がいたら | 家の記録・予定・メモ |
| 3 | 特定の誰かがいたら | その人への申し送り（器は `pending_speech.target_person_id`） |

段は**きっかけの札で一律に決める**（本人の決定）。中身ごとに主LLM へ決めさせる案もあるが、
そちらは測ってからにする。一律なら揺れず、`say()` にも `pending_speech` にも欄を足さずに済む。
"""

from __future__ import annotations

#: 誰もいなくても言う。
ALONE = 0
#: 誰かいれば言う（誰かは分からなくてよい）。既定。
ANYONE = 1
#: 家族と確かめられた人が居れば言う。
FAMILY = 2
#: その人が居れば言う。
SOMEONE = 3

#: きっかけの札 → 段。**ここに無い札は既定（誰かいたら）**——厳しい側へ倒すと、
#: 札が増えたときに黙って届かなくなる。
_BY_LABEL: "dict[str, int]" = {
    "メモ": FAMILY,  # 家の記録（Obsidian のメモ）。中身は家族の予定・決めごと
}


def level_of(request_text: str) -> int:
    """きっかけの本文（`[メモ] …`）から宛先の条件を返す。札が無ければ既定。"""
    s = (request_text or "").lstrip()
    if not s.startswith("["):
        return ANYONE
    label = s[1 : s.find("]")] if "]" in s else ""
    return _BY_LABEL.get(label.strip(), ANYONE)


def meets(level: int, presence_rows: "list[dict]") -> bool:
    """いまの在席が、その段を満たすか。

    `presence_rows` は `PersonMemoryManager.presence_status()` の行。**名前の分からない
    在席者は `person_id` が `None`**（出-ae-は の札）なので、段 2 は満たさない——家族と
    確かめられていない相手に、家の記録は渡さない。
    """
    if level <= ALONE:
        return True
    rows = list(presence_rows or ())
    if not rows:
        return False
    if level == ANYONE:
        return True
    return any(r.get("person_id") for r in rows)
