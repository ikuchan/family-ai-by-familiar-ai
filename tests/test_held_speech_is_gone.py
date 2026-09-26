"""保留して後で配る仕組みと宛先の条件は外した（出-as 段 9b・2026-09-26・`設計方針_話していいかの決まり` §2.6）。

止められた発話はすべて独り言にする（段 6）。保留を積む口・配る口・W の絞り込み（出-ap）・退室の分岐（知-r・
退室は記録だけになった）・「誰も見えなかった」の控え（入口で不在を止めなくなった）は、もう通る道が無い。
名前が残ると、次に読む者が動いている仕組みだと信じるので、src から消えたことを確かめる。

`pending_speech` テーブルとその店（`tools/pending_speech_store.py`）は残す（落とすのは 環-ab）。
"""

from __future__ import annotations

import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "familiar_agent"

GONE = (
    "_hold_speech",
    "_release_pending_speech",
    "_HELD_REASON_PAST",
    "speech_to_deliver",
    "release_pending",
    "core.audience",
    "_may_show",
    "誰も見えなかった",
    "誰も見えないあいだに聞いた",
)


def test_the_old_names_are_gone_from_src():
    hits = []
    for p in SRC.rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        for name in GONE:
            if name in text:
                hits.append(f"{p.relative_to(SRC)}: {name}")
    assert hits == []


def test_the_audience_module_is_gone():
    assert not (SRC / "core" / "audience.py").exists()


def test_speak_has_no_branch_for_leaving():
    src = (SRC / "loop" / "event_loop.py").read_text(encoding="utf-8")
    body = src[src.index("async def _speak(") : src.index("async def _say_filler(")]
    assert not re.search(r"\[退室\]", body)
