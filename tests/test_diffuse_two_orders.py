"""関連想起（拡散）は 2 つの並びから交互に取る（記-a-ろ-い・2026-09-14）。

- **遠い順**：種のベクトルから分類上遠いもの（新規性・4b・既存）
- **思い出していない順**：`last_recalled_at` が古いもの（無ければ最古扱い）

4 枠を 2／2 で交互に埋め、同じ記録は 1 件と数える。配分は層 3 の設定値 `diffuse_far_share`。
`last_recalled_at` は想起のたびに更新されるので（`touch_recalled`）、この並びは「長く
思い出していないもの」を掘り起こす。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from familiar_agent.core.diffuse import interleave_orders


def test_two_orders_are_interleaved_and_deduplicated():
    far = ["a", "b", "c", "d"]
    stale = ["b", "e", "a", "f"]
    assert interleave_orders(far, stale, max_add=4, far_share=0.5) == ["a", "b", "c", "e"]


def test_far_share_decides_how_many_come_from_each_order():
    far = ["a", "b", "c", "d"]
    stale = ["e", "f", "g", "h"]
    assert interleave_orders(far, stale, max_add=4, far_share=0.75) == ["a", "e", "b", "c"]
    assert interleave_orders(far, stale, max_add=4, far_share=0.0) == ["e", "f", "g", "h"]


def test_a_short_order_lets_the_other_fill_the_rest():
    assert interleave_orders(["a"], ["b", "c", "d", "e"], max_add=4, far_share=0.5) == [
        "a",
        "b",
        "c",
        "d",
    ]


def test_stalest_first_puts_never_recalled_before_old_and_old_before_recent():
    from familiar_agent.diffuse_store import order_ids_by_stalest_rows

    now = datetime.now(timezone.utc)
    rows = {
        "recent": now - timedelta(hours=1),
        "old": now - timedelta(days=20),
        "never": None,
    }
    assert order_ids_by_stalest_rows(["recent", "old", "never"], rows) == ["never", "old", "recent"]
