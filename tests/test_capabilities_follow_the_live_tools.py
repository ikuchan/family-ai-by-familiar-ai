"""能力は、いま繋がっている道具で数える（出-al・2026-09-22）。

実機 17:50、カレンダーの MCP を外した状態で「明日の予定わかる？」と聞くと、3 回続けて
「調べたけど出てこない」と言い、**道具が無いことには一度も触れなかった**。

そのとき、能力の門は `enabled_env: MCP_CONFIG` だった。この機体は `MCP_CONFIG` を設定せず
既定パス `~/.familiar-ai.json` を使うので、**MCP 由来の能力は、使えていても一覧から落ちる**。
実態と逆にずれていた。門を**道具の名前**へ変え、その道具がいま取れるときだけ有効と数える。
"""

from __future__ import annotations

import pathlib

from familiar_agent.capability_state import collect_manifest_context, filter_enabled

_MANIFEST = """capabilities:
  - id: family_schedule
    summary: Look up the family's schedule
    detail: >
      `get_family_schedule` returns what is on the family calendar.
    enabled_tool: get_family_schedule
  - id: camera_vision
    summary: See through a camera
    detail: >
      The camera.
    enabled_env: CAMERA_HOST
  - id: memory
    summary: Remember
    detail: >
      Memory.
    enabled: true
"""


# ── 道具の名前で門を作る ───────────────────────────────────────────────────


def test_a_capability_is_kept_when_its_tool_is_there():
    out = filter_enabled(_MANIFEST, env={}, tools={"get_family_schedule"})
    assert "family_schedule" in out


def test_a_capability_is_dropped_when_its_tool_is_gone():
    out = filter_enabled(_MANIFEST, env={}, tools=set())
    assert "family_schedule" not in out, "繋がっていない道具を能力として語っている"


def test_unknown_tools_do_not_touch_the_other_gates():
    """`enabled` と `enabled_env` はそのまま効く（カメラ・記憶はこちらの門）。"""
    out = filter_enabled(_MANIFEST, env={"CAMERA_HOST": "1"}, tools=set())
    assert "camera_vision" in out
    assert "memory" in out


def test_no_tools_given_means_no_tool_gated_capability():
    """道具の一覧を渡さない呼び方では、道具の門を持つ能力は残さない。

    **知らないときは語らない**——渡し忘れた側が「使える」と嘘をつくより、落ちて気づくほうがよい。
    """
    out = filter_enabled(_MANIFEST, env={"CAMERA_HOST": "1"})
    assert "family_schedule" not in out
    assert "camera_vision" in out


# ── 実物の一覧が道具の門を使う ────────────────────────────────────────────


def test_the_shipped_manifest_gates_mcp_capabilities_by_tool():
    text = pathlib.Path("capabilities.yaml").read_text(encoding="utf-8")
    assert "enabled_tool: get_family_schedule" in text
    assert "enabled_tool: get_house_rules" in text
    assert "enabled_tool: search_notion" in text


# ── 層 4 の材料に、いま繋がっている道具が入る ────────────────────────────


def test_the_material_names_the_live_tools():
    ctx = collect_manifest_context(live_tools=["get_family_schedule", "search_notion"])
    assert "get_family_schedule" in ctx
    assert "search_notion" in ctx


def test_the_material_says_so_when_nothing_is_connected():
    ctx = collect_manifest_context(live_tools=[])
    assert "## Live tools" in ctx, "いま取れる道具の節が要る"
