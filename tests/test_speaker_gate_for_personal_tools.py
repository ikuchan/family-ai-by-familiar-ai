"""話者ゲート（知-f・`設計方針_家の記録との接続` §3）。

個人ティアの道具は名前に人が入っている（`ask_vault_yusuke`）。**その人のターン以外では
道具の定義そのものを出さない**——description で頼むのでなく構造で落とす。人と英字の対応は
FAMILY.md の `- **英字**：Yusuke Ikunaga`（先頭の語を小文字にした `yusuke` が鍵）。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from familiar_agent.core.parsing import parse_family_md
from familiar_agent.core.tool_gate import gate_personal_tools

FAMILY = """# 一緒に暮らす人たち

## ゆうすけ

- **名前**：ゆうすけ
- **呼び方**：パパ
- **英字**：Yusuke Ikunaga

## たいき

- **名前**：たいき
- **呼び方**：たいき
- **英字**：Taiki Ikunaga

## こうき

- **名前**：こうき
- **呼び方**：こうき
"""

MEMBERS = parse_family_md(FAMILY)


def _defs(*names: str) -> list[dict]:
    return [{"name": n, "description": "…", "input_schema": {"type": "object"}} for n in names]


# ---- FAMILY.md の英字 ----------------------------------------------------------


def test_the_latin_name_is_the_lowercased_first_word() -> None:
    by_name = {m["name"]: m for m in MEMBERS}
    assert by_name["ゆうすけ"]["latin"] == "yusuke"
    assert by_name["たいき"]["latin"] == "taiki"
    assert by_name["こうき"]["latin"] == ""  # 書いていない人は個人ティアの道具を持たない


def test_a_template_placeholder_is_not_a_latin_name() -> None:
    text = "## 例\n- **名前**：田中太郎\n- **英字**：（ローマ字の名前 — 例：Taro Tanaka）\n"
    assert parse_family_md(text)[0]["latin"] == ""


# ---- ゲート ---------------------------------------------------------------------


def test_the_owner_keeps_the_personal_tool_and_others_lose_it() -> None:
    defs = _defs("get_house_rules", "ask_vault_yusuke", "search_notion")
    kept = [d["name"] for d in gate_personal_tools(defs, speaker="ゆうすけ", members=MEMBERS)]
    assert kept == ["get_house_rules", "ask_vault_yusuke", "search_notion"]
    kept = [d["name"] for d in gate_personal_tools(defs, speaker="たいき", members=MEMBERS)]
    assert kept == ["get_house_rules", "search_notion"]


def test_an_unknown_speaker_gets_no_personal_tool() -> None:
    defs = _defs("get_house_rules", "ask_vault_yusuke")
    kept = [d["name"] for d in gate_personal_tools(defs, speaker="", members=MEMBERS)]
    assert kept == ["get_house_rules"]


def test_a_suffix_that_is_nobody_is_not_a_person() -> None:
    # 人の英字で終わらない名前は家族ティア（`get_family_schedule` の `_schedule` を人と見ない）。
    defs = _defs("get_family_schedule", "ask_vault_zzz")
    kept = [d["name"] for d in gate_personal_tools(defs, speaker="たいき", members=MEMBERS)]
    assert kept == ["get_family_schedule", "ask_vault_zzz"]


def test_the_gate_matches_a_whole_suffix_not_a_substring() -> None:
    # `_taiki` が `_ki` や `iki` に釣られない。区切りは `_` で、末尾まで一致したときだけ。
    members = parse_family_md("## き\n- **名前**：き\n- **英字**：Ki X\n")
    defs = _defs("ask_vault_taiki")
    assert [d["name"] for d in gate_personal_tools(defs, speaker="き", members=members)] == [
        "ask_vault_taiki"
    ]


# ---- ループの出口で効く ---------------------------------------------------------


def _ip_with(tool_names: list[str], speaker: str):
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    a._family_md = FAMILY
    ip = InformationProcessing(a)
    ip._current_speaker_name = lambda: speaker  # type: ignore[method-assign]
    ip._ACTIONS = dict(ip._ACTIONS)
    ip._ACTIONS["vault"] = lambda _ip: _defs(*tool_names)
    ip._dif = MagicMock()
    ip._dif.tool_defs = lambda name: _defs(name)
    return ip


def test_the_main_llm_tools_pass_through_the_gate() -> None:
    ip = _ip_with(["ask_vault_yusuke"], speaker="ゆうすけ")
    assert [d["name"] for d in ip._tools(actions=("vault",))] == ["ask_vault_yusuke"]
    ip = _ip_with(["ask_vault_yusuke"], speaker="たいき")
    assert ip._tools(actions=("vault",)) == []


def test_the_arbiter_candidates_pass_through_the_same_gate() -> None:
    ip = _ip_with([], speaker="たいき")
    ip._ACTIONS["house_rules"] = lambda _ip: _defs("ask_vault_yusuke")  # 個人ティアを装う
    assert "house_rules" not in ip._extra_actions()
    ip._current_speaker_name = lambda: "ゆうすけ"  # type: ignore[method-assign]
    assert "house_rules" in ip._extra_actions()
