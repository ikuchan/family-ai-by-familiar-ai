"""沈黙依頼の機械の守り（2026-09-16 実機）。

読むのは調停（軽量LLM）だが、読みだけに任せると両方向で外れた：「待てぃ」「しなよ」（STT の
断片）を沈黙の依頼と読んで 60 分黙り、「話していいよ」を解除と読まなかった。設計（名前で呼ばれた
ときだけ受ける）を、プロンプトの指示でなく機械で守る。

- `names_me`：発話に自分の名前（`ME.md` の名前・呼び方）が含まれるか。掛ける側の条件。
- `is_release`：黙っているあいだに「話していい」と言われたか。解く側の条件（調停の返りに依らない）。
- `silence_note`：調停へ渡す「いま黙っている」の一行。黙っている前提が無いと「解かれた」と読めない。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..silence_state import SilenceRequest

_RELEASE = re.compile(r"(話|はな|しゃべ|喋)(し|っ)?て(も)?(いい|良い|よい)")


def names_me(utterance: str, names: "list[str] | tuple[str, ...]") -> bool:
    """発話に自分の名前のどれかが入っているか（部分一致・大文字小文字は無視）。"""
    text = (utterance or "").lower()
    return any(n and n.lower() in text for n in names)


def is_release(utterance: str) -> bool:
    """「話していいよ」「しゃべってもいい」の類か。"""
    return bool(_RELEASE.search(utterance or ""))


def silence_note(req: "SilenceRequest | None", *, now: float) -> str:
    """調停の `[いま]` に添える一行。依頼が無い・切れているなら空。"""
    if req is None or now >= req.until:
        return ""
    left = max(1, int((req.until - now) // 60))
    return f"黙っているよう頼まれている（{req.person}から・あと約 {left} 分）"
