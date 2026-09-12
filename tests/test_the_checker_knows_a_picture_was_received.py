"""整合チェックの材料に、画像を受け取ったことと写っていたものを入れる（`イベント駆動ループ` v0.43）。

軽量LLM は画像を持たない。渡さないと、主LLM が写真を見て正しく語った返事を
「与えられていない」と差し戻す（2026-09-12 実機）。
"""

from __future__ import annotations

from familiar_agent.loop.coherence import facts_ctx


def test_the_picture_and_the_mark_are_in_the_facts() -> None:
    ctx = facts_ctx(
        saw=True, memories=[], picture=True, seen="出入り口を見た。見えたもの：person、chair"
    )
    assert "画像を受け取った：はい" in ctx
    assert "person、chair" in ctx


def test_without_a_picture_the_facts_say_so() -> None:
    ctx = facts_ctx(saw=False, memories=[])
    assert "画像を受け取った：いいえ" in ctx
