"""口調を、いま向き合っている相手で決める（出-ak・2026-09-23）。純関数。

実機 17:25、話者が「パパ」で確定しているのに常体で話した。`ME.md` には「大人（パパ・ママ）
には〜です〜ます」とあり、`FAMILY.md` にも「大人」とある。**規則は届いていて、守られなかった。**

**口調の正本は `ME.md` のままにする。** 機械は写しを持たず、`ME.md` の話し方から**その相手
向けの行をそのまま抜く**だけにする。写しを持つと、`ME.md` を書き換えても機械の行が古いまま
残る。実測では、`ME.md` を「大人にも常体」に書き換えれば常体になった（正本が効いている）。

抜いた行の言い回しは効き目に効く。「〜を基本にする」では 12 回中 6 回、「〜で話す」では
12 回中 11 回が敬体になった。**含みのある言い方は弱い。**
"""

from __future__ import annotations

from . import parsing

#: 話し方の行の頭。`ME.md` の「話し方：」の下に、相手ごとの行が並ぶ。
_ADULT = "大人"
_CHILD = "子ども"


def tone_line(me_md: str, *, adult: bool) -> str:
    """`ME.md` の話し方から、その相手向けの行をそのまま返す。無ければ空。

    行頭の `- ` と前後の空白だけを落とす。**中身は変えない**——正本の言葉をそのまま渡す。
    """
    want = _ADULT if adult else _CHILD
    for raw in (me_md or "").splitlines():
        line = raw.strip().lstrip("-").strip()
        if line.startswith(want):
            return line
    return ""


def is_adult(name: str, family_md: str) -> "bool | None":
    """その人が大人か。`FAMILY.md` の「関係」で判じる。書いていなければ `None`。

    `None` は「分からない」であって「子ども」ではない。**決めつけない**——`ME.md` の
    「相手が分からないときは、大人として扱い丁寧に話す」は、呼び手の側の決まりである。
    """
    s = (name or "").strip()
    if not s:
        return None
    for m in parsing.parse_family_md(family_md or ""):
        if s not in _aliases(m):
            continue
        rel = str(m.get("relation") or "")
        if _ADULT in rel:
            return True
        if _CHILD in rel:
            return False
        return None
    return None


def _aliases(member: dict) -> "list[str]":
    """その人を指す言い方を全部（名前＋呼び方の一つずつ）。"""
    from .speaker_claim import aliases_of

    return aliases_of(member)
