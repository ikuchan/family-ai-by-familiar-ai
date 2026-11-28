"""出-dy：返事は、今日から数えるもの（日数・曜日）を検索の返りのページから写さず、いまの日時から数える。

2026-11-28 20:17「クリスマスまであと何日?」に、検索の返りの写し（11/25 ごろの「There are 30 days until Christmas」）を
そのまま使い「あと30日」と答えた（正しくは 27 日）。検索のページの「あと N 日」は、書かれた日（写された日）の数である。
返事は軽量LLM（調停の文章）も主LLM も書くので、両方の決まりに置く。
"""

from __future__ import annotations

from familiar_agent.loop.arbiter import _FIELD_TEXT, _HEADING_TOOL_RETURN
from familiar_agent.loop.prompt import EVENT_SYSTEM_PROMPT


def test_the_light_reply_counts_days_from_now_not_from_the_tool_return():
    text = _FIELD_TEXT["text"]
    assert "[いま] から数える" in text and "書かれた日" in text
    assert _HEADING_TOOL_RETURN in text  # 検索の返りが並ぶ見出しの名で指す


def test_the_main_llm_counts_days_from_now_too():
    assert ":id count-from-now" in EVENT_SYSTEM_PROMPT
    assert "now から数える" in EVENT_SYSTEM_PROMPT


def test_other_light_fields_are_unchanged():
    """**反証の側**：問い・話しかけ・自分からの一言などの欄には足さない（返事の欄だけ）。"""
    for key, text in _FIELD_TEXT.items():
        if key != "text":
            assert "書かれた日" not in text, key
