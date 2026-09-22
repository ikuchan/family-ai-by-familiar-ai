"""名乗りで話者を付ける（知-w・2026-09-19）。純関数。

「パジュ、こんにちはパパだよ」が、カメラに映らない位置からの声で入口に飲まれた（実機 11:32）。在席（居るか）は
カメラだけで決める（マイクは証拠にしない）。**カメラが人を見ているとき**だけ、声の名乗り（「パパだよ」「僕は
たいき」）から話者を付ける（`/speaker` と同じ効き）。名乗りの読みは調停（`Decision.speaker_claim`）、ここは検め：
家族（`FAMILY.md`）の名前か呼び方に一致すれば、**呼びかけに使う名前**（呼び方の先頭）を返す。無ければ None。

**呼び方は一つずつ照合する**（2026-09-22 に直した）。以前は一覧まるごと
（"パパ、ゆうすけ、おとうさん、Papa、father"）としか比べておらず、いちばん自然な「パパだよ」「ママだよ」で
話者が付かなかった——当たるのは `名前` の欄と一致する言い方だけだった。返り値も一覧のままで、
`set_active()` にそれが入り、在席の文脈が `(present :speaker "パパ、ゆうすけ、…")` になっていた。
"""

from __future__ import annotations

from . import parsing

#: 呼び方の区切り（`FAMILY.md` の「呼び方」は読点区切りの一覧）。
_SEPARATORS = ("、", ",")


def aliases_of(member: dict) -> "list[str]":
    """その人を指す言い方を全部（名前＋呼び方の一つずつ）。空は落とす。"""
    out: list[str] = []
    name = str(member.get("name") or "").strip()
    if name:
        out.append(name)
    display = str(member.get("display_name") or "").strip()
    parts = [display]
    for sep in _SEPARATORS:
        parts = [q for p in parts for q in p.split(sep)]
    for p in parts:
        p = p.strip()
        if p and p not in out:
            out.append(p)
    return out


def call_name_of(member: dict) -> str:
    """呼びかけに使う名前。呼び方の先頭、無ければ名前。"""
    display = str(member.get("display_name") or "").strip()
    for sep in _SEPARATORS:
        display = display.split(sep)[0]
    return display.strip() or str(member.get("name") or "").strip()


def resolve_claim(claim: str, family_md: str) -> "str | None":
    s = (claim or "").strip(" \t　「」")
    if not s:
        return None
    for m in parsing.parse_family_md(family_md or ""):
        if s in aliases_of(m):
            return call_name_of(m) or None
    return None
