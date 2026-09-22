"""使えない道具を人に伝える（出-al・2026-09-22）。

実機 17:50、カレンダーの MCP が設定から外れた状態で「明日の予定わかる？」と聞くと、3 回
続けて「調べたけど出てこない」と言い、**道具が無いことには一度も触れなかった**。繋がって
いない道具は `DIF.tool_defs()` が空を返して**黙って消える**ので、無いという事実がどこにも
書かれていなかった。

実機の作業状態と、カレンダー抜きの道具一式で測ると、`[いま使えない]` の 1 行を置くだけで
8 回中 8 回が伝える（置かないと 0 回で、8 回とも `search_notion` を選び直す）。**言い聞かせる
文は要らない**——「聞かれたら正直に言う」を足しても 8 回のままだった。
"""

from __future__ import annotations

from familiar_agent.capability_state import missing_tools
from familiar_agent.loop.generator import _iter_ctx

_MANIFEST = """capabilities:
  - id: family_schedule
    summary: Look up the family's schedule
    detail: >
      `get_family_schedule` returns the calendar.
    enabled_tool: get_family_schedule
  - id: house_rules
    summary: Look up the rules
    detail: >
      `get_house_rules` returns the rules.
    enabled_tool: get_house_rules
  - id: memory
    summary: Remember
    detail: >
      Memory.
    enabled: true
"""


# ── 何が欠けているかを数える ──────────────────────────────────────────────


def test_a_tool_that_is_not_reachable_is_missing():
    got = missing_tools(_MANIFEST, {"get_house_rules"})
    assert [t for t, _ in got] == ["get_family_schedule"]


def test_nothing_is_missing_when_everything_is_there():
    assert missing_tools(_MANIFEST, {"get_family_schedule", "get_house_rules"}) == []


def test_a_capability_without_a_tool_gate_is_never_missing():
    """`enabled: true` の能力は道具の門を持たないので、ここには出てこない。"""
    got = missing_tools(_MANIFEST, set())
    assert "memory" not in [n for _, n in got]


def test_the_missing_tool_has_a_name_people_understand():
    got = missing_tools(_MANIFEST, set())
    names = dict(got)
    assert "カレンダー" in names["get_family_schedule"]
    assert "決まり" in names["get_house_rules"]


# ── 反復の行の手前に置く ──────────────────────────────────────────────────


def test_the_line_comes_before_the_reply_budget():
    from familiar_agent.loop import reply_budget

    text = _iter_ctx(
        chain=2,
        max_chain=5,
        thinking_round=1,
        capped=False,
        budget=reply_budget.decide(effort="low", researched=True, w_count=3),
        missing=["家族の予定を見る道具（カレンダー）"],
    )
    assert text.startswith("[いま使えない] 家族の予定を見る道具（カレンダー）")
    assert text.index("[いま使えない]") < text.index("[返事]")


def test_nothing_is_added_when_nothing_is_missing():
    text = _iter_ctx(chain=1, max_chain=5, thinking_round=1, capped=False, missing=[])
    assert "[いま使えない]" not in text


def test_several_missing_tools_are_listed_on_one_line():
    text = _iter_ctx(
        chain=1,
        max_chain=5,
        thinking_round=1,
        capped=False,
        missing=["家族の予定を見る道具（カレンダー）", "家の決まりを見る道具"],
    )
    first = text.splitlines()[0]
    assert "カレンダー" in first and "決まり" in first
