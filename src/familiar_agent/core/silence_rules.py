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
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..silence_state import SilenceRequest

_RELEASE = re.compile(r"(話|はな|しゃべ|喋)(し|っ)?て(も)?(いい|良い|よい)")


_SMALL_KANA = str.maketrans("ぁぃぅぇぉゃゅょっゎ", "あいうえおやゆよつわ")


def _loose(s: str) -> str:
    """名前を比べるための**ゆるい読み**。

    STT は「パジュ」を はじゅ／パチュー と書く（2026-09-17 実機）。カタカナはひらがなに、
    長音は落とし、小書きは大きく、濁点・半濁点は落とす（NFD で分解して結合記号を捨てる）。
    """
    out = []
    for ch in unicodedata.normalize("NFD", (s or "").lower()):
        if ch in ("\u3099", "\u309a", "ー", "〜", "～"):
            continue
        o = ord(ch)
        if 0x30A1 <= o <= 0x30F6:  # カタカナ → ひらがな
            ch = chr(o - 0x60)
        out.append(ch)
    return "".join(out).translate(_SMALL_KANA)


def _within_one_edit(a: str, b: str) -> bool:
    """同じ長さの 2 語が 1 文字の置き換え以内か（挿入・削除は窓の長さで別に見る）。"""
    return sum(x != y for x, y in zip(a, b)) <= 1


def names_me(utterance: str, names: "list[str] | tuple[str, ...]") -> bool:
    """発話に自分の名前のどれかが入っているか。

    ゆるい読み（`_loose`）で比べ、名前が 3 文字以上なら **1 文字違いまで**許す（置き換え 1 つ、
    または 1 文字の抜け・足し）。2 文字以下は何にでも当たるので完全一致のまま。
    """
    text = _loose(utterance)
    for raw in names:
        n = _loose(raw)
        if not n:
            continue
        if n in text:
            return True
        if len(n) < 3:
            continue
        for width in (len(n), len(n) - 1, len(n) + 1):
            for i in range(0, max(0, len(text) - width) + 1):
                w = text[i : i + width]
                if len(w) != width:
                    continue
                if width == len(n) and _within_one_edit(w, n):
                    return True
                if width != len(n) and _one_insertion_apart(w, n):
                    return True
    return False


def _one_insertion_apart(a: str, b: str) -> bool:
    """長さが 1 違う 2 語が、1 文字の挿入だけで一致するか。"""
    if len(a) > len(b):
        a, b = b, a
    i = j = 0
    skipped = False
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        elif not skipped:
            skipped = True
            j += 1
        else:
            return False
    return True


def is_release(utterance: str) -> bool:
    """「話していいよ」「しゃべってもいい」の類か。"""
    return bool(_RELEASE.search(utterance or ""))


def silence_note(req: "SilenceRequest | None", *, now: float) -> str:
    """調停の `[いま]` に添える一行。依頼が無い・切れているなら空。"""
    if req is None or now >= req.until:
        return ""
    left = max(1, int((req.until - now) // 60))
    if getattr(req, "reason", "").startswith("timer:"):
        return f"タイマーが鳴るまで黙っている（{req.person}が掛けた・あと約 {left} 分）"
    return f"黙っているよう頼まれている（{req.person}から・あと約 {left} 分）"
