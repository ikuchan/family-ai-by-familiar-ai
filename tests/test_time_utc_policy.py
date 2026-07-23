"""時刻方針：DB は UTC で管理・プロンプトは OS タイムゾーン付記（Slice 1）。

naive `datetime.utcnow()` を aware UTC（`clock.now_utc()`／`now_utc_iso()`）へ統一し、
プロンプトの現在時刻は OS ローカル＋tz 付き（`now_local_str()`）にする。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from familiar_agent.store import clock


def test_now_utc_iso_is_aware_utc():
    dt = datetime.fromisoformat(clock.now_utc_iso())
    assert dt.tzinfo is not None
    assert dt.utcoffset() == timezone.utc.utcoffset(None)  # +00:00


def test_now_local_str_has_timezone_suffix():
    s = clock.now_local_str()
    # OS のオフセット（例 +0900）が付記されている。
    assert re.search(r"[+-]\d{4}\)?$", s), s
    assert s[:4].isdigit()  # 先頭は年


def test_working_memory_item_created_at_is_aware_utc():
    from familiar_agent.mental_state import WorkingMemoryItem

    item = WorkingMemoryItem(memory_id="x", summary="s", source_kind="k", salience=0.5)
    dt = datetime.fromisoformat(item.created_at)
    assert dt.tzinfo is not None  # naive utcnow ではなく aware


def test_no_naive_utcnow_remains_in_src():
    """完了条件：src に naive `utcnow()` が残っていない（grep 0）。"""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "src" / "familiar_agent"
    hits = [
        f"{p}"
        for p in root.rglob("*.py")
        if "utcnow()" in p.read_text(encoding="utf-8")
    ]
    assert hits == [], f"naive utcnow() remains: {hits}"
