"""名乗りで話者を付ける（知-w・2026-09-19）。純関数。

「パジュ、こんにちはパパだよ」が、カメラに映らない位置からの声で入口に飲まれた（実機 11:32）。在席（居るか）は
カメラだけで決める（マイクは証拠にしない）。**カメラが人を見ているとき**だけ、声の名乗り（「パパだよ」「僕は
たいき」）から話者を付ける（`/speaker` と同じ効き）。名乗りの読みは調停（`Decision.speaker_claim`）、ここは検め：
家族（`FAMILY.md`）の名前か呼び方に一致すれば呼び方（`/speaker` と同じ表記）を返す。無ければ None。
"""

from __future__ import annotations

from . import parsing


def resolve_claim(claim: str, family_md: str) -> "str | None":
    s = (claim or "").strip(" \t　「」")
    if not s:
        return None
    for m in parsing.parse_family_md(family_md or ""):
        name = str(m.get("name") or "").strip()
        display = str(m.get("display_name") or "").strip()
        if s in (name, display) and (display or name):
            return display or name
    return None
