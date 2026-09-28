"""vulture の許可リストは、いま在る名前だけを持つ（環-ab R-7c・2026-09-28）。

消した名前が許可リストに残ると、同じ名前を新しく書いたときに「使われていない」が出なくなる。
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_every_whitelisted_name_still_exists_in_src():
    names = re.findall(
        r"^(?:_\.)?(\w+)  #", (ROOT / "vulture_whitelist.py").read_text(encoding="utf-8"), re.M
    )
    assert names, "許可リストから名前が読めない（書式が変わった）"
    src = "\n".join(
        p.read_text(encoding="utf-8") for p in (ROOT / "src" / "familiar_agent").rglob("*.py")
    )
    assert [n for n in names if not re.search(rf"\b{re.escape(n)}\b", src)] == []
