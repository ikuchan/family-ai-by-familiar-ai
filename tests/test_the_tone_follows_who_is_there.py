"""口調は、いま向き合っている相手で決まる（出-ak・2026-09-23）。

実機 17:25〜17:26、`/speaker パパ` で話者が確定している状態で「…分からないんだ。ごめんね、
パパ！」と常体だった。`ME.md` には「大人（パパ・ママ）には〜です〜ます」とあり、`FAMILY.md`
にも「大人」とある。**規則は届いていて、守られなかった。**

実機の作業状態で 8 通り測った結果、効いたのは次の 3 つの組み合わせだけだった（単独では
相 1/12・返 2/12・在 0/12、3 つで **11/12**）。

- 相：W の過去の発話に**誰へ言ったか**を添える（子どもへの常体が大人との会話に並んでいた）
- 在：在席の行に「この相手は大人」
- 返：`[返事]` の行に「いま向き合っている相手は大人。」＋ `ME.md` の話し方の行

**口調の正本は `ME.md` のまま。** 機械は写しを持たず、行を抜いて当てる先を言うだけにする。
`ME.md` を「大人にも常体」に書き換えると、そのとおり常体になることも確かめた。
"""

from __future__ import annotations

from familiar_agent.core.tone import is_adult, tone_line

_ME = """# パジュについて

名前：パジュ
話し方：標準語。丁寧さは相手で決め、一つの返事の中では混ぜない。
　　　　- 大人（パパ・ママ）には「〜です」「〜ます」で話す。
　　　　- 子ども（たいき・こうき）には「〜だよ」「〜だね」と打ち解けて話す。
　　　　- 相手が分からないときは、大人として扱い丁寧に話す。
一人称：ぼく
"""

_FAMILY = """## ゆうすけ

- **名前**：ゆうすけ
- **呼び方**：パパ、ゆうすけ、おとうさん
- **関係**：家族の父。大人。パジュを作った人

## たいき

- **名前**：たいき
- **呼び方**：たいき、たいきくん
- **関係**：お兄ちゃん（長男）。子ども。パジュを作るのを手伝った

## なぞ

- **名前**：なぞ
- **呼び方**：なぞ
- **関係**：（書いていない）
"""


# ── ME.md から、その相手向けの行を抜く ───────────────────────────────────


def test_the_line_for_adults_is_taken_as_written():
    assert tone_line(_ME, adult=True) == "大人（パパ・ママ）には「〜です」「〜ます」で話す。"


def test_the_line_for_children_is_taken_as_written():
    assert (
        tone_line(_ME, adult=False)
        == "子ども（たいき・こうき）には「〜だよ」「〜だね」と打ち解けて話す。"
    )


def test_rewriting_me_md_changes_what_is_taken():
    """正本は `ME.md`。機械は写しを持たない。"""
    rewritten = _ME.replace(
        "大人（パパ・ママ）には「〜です」「〜ます」で話す。",
        "大人（パパ・ママ）にも「〜だよ」「〜だね」と打ち解けて話す。",
    )
    assert "だよ" in tone_line(rewritten, adult=True)


def test_nothing_is_taken_when_me_md_says_nothing():
    assert tone_line("名前：パジュ\n一人称：ぼく\n", adult=True) == ""
    assert tone_line("", adult=True) == ""


# ── FAMILY.md から、大人かを判じる ───────────────────────────────────────


def test_an_adult_is_recognised_by_any_of_their_names():
    assert is_adult("パパ", _FAMILY) is True
    assert is_adult("ゆうすけ", _FAMILY) is True
    assert is_adult("おとうさん", _FAMILY) is True


def test_a_child_is_not_an_adult():
    assert is_adult("たいき", _FAMILY) is False
    assert is_adult("たいきくん", _FAMILY) is False


def test_someone_without_a_relation_is_unknown():
    """書いていない人は**分からない**（None）。大人と決めつけない。"""
    assert is_adult("なぞ", _FAMILY) is None


def test_someone_outside_the_family_is_unknown():
    assert is_adult("太郎", _FAMILY) is None
    assert is_adult("", _FAMILY) is None


# ── FAMILY.md の解析が「関係」を持つ ─────────────────────────────────────


def test_the_family_parser_keeps_the_relation():
    """`関係` は、大人か子どもかを判じる唯一の材料。落とすと口調が決まらない。"""
    from familiar_agent.core import parsing

    got = {m["name"]: m for m in parsing.parse_family_md(_FAMILY)}
    assert "大人" in got["ゆうすけ"]["relation"]
    assert "子ども" in got["たいき"]["relation"]


def test_a_missing_relation_is_empty_not_absent():
    from familiar_agent.core import parsing

    got = {m["name"]: m for m in parsing.parse_family_md("## x\n- **名前**：x\n")}
    assert got["x"]["relation"] == ""


# ── W の行に「誰へ言ったか」が入る ───────────────────────────────────────


class _OIF:
    def __init__(self, addressees=None):
        self._addr = addressees or {}

    def voices(self, ids):
        return {i: ("わたし", self._addr.get(i, [])) for i in ids if i in self._addr}


def _row(obs_id: str, role: str, content: str):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    return SimpleNamespace(
        obs_id=obs_id,
        role=role,
        content=content,
        when=datetime(2026, 9, 21, 17, 25, tzinfo=timezone.utc),
        direction="発話",
    )


def test_my_own_line_says_who_i_spoke_to():
    from familiar_agent.loop import workspace

    rows = [_row("o1", "答え", "自分が答えた：おかえりなさい")]
    _, text, _ = workspace.render_recent(_OIF({"o1": ["パパ"]}), [("o1", rows)], 3)
    assert "わたし（パパへ）：" in text


def test_several_addressees_are_listed():
    from familiar_agent.loop import workspace

    rows = [_row("o1", "答え", "自分が答えた：おかえり！")]
    _, text, _ = workspace.render_recent(_OIF({"o1": ["たいき", "こうき"]}), [("o1", rows)], 3)
    assert "わたし（たいき・こうきへ）：" in text


def test_nothing_is_added_when_the_addressee_is_unknown():
    """相手を捏造しない。面が立っていなければ、いままでどおり `わたし：`。"""
    from familiar_agent.loop import workspace

    rows = [_row("o1", "答え", "自分が答えた：ひとりごと")]
    _, text, _ = workspace.render_recent(_OIF({}), [("o1", rows)], 3)
    assert "わたし：" in text
    assert "（" not in text.split("わたし")[1][:3]


def test_the_other_persons_line_is_unchanged():
    from familiar_agent.loop import workspace

    rows = [_row("o1", "起点", "おかえり")]
    _, text, _ = workspace.render_recent(_OIF({"o1": ["パパ"]}), [("o1", rows)], 3)
    assert "相手：" in text


# ── 在席の行に「この相手は大人」 ─────────────────────────────────────────


def _agent_with(rows, family=_FAMILY):
    class _PMM:
        def presence_status(self):
            return rows

    class _Agent:
        _pmm = _PMM()
        _family_md = family

    return _Agent()


def test_an_adult_speaker_is_marked():
    from familiar_agent.loop.generator import _present_ctx

    rows = [{"person_id": "p1", "name": "パパ", "confidence": 1.0, "is_speaker": True}]
    assert "この相手は大人" in _present_ctx(_agent_with(rows))


def test_a_child_speaker_is_not_marked_as_adult():
    from familiar_agent.loop.generator import _present_ctx

    rows = [{"person_id": "p1", "name": "たいき", "confidence": 1.0, "is_speaker": True}]
    assert "この相手は大人" not in _present_ctx(_agent_with(rows))


# ── [返事] の行に、ME.md の口調の行が付く ────────────────────────────────


def test_the_reply_line_carries_the_tone_from_me_md():
    from familiar_agent.loop import reply_budget
    from familiar_agent.loop.generator import _iter_ctx

    text = _iter_ctx(
        chain=1,
        max_chain=5,
        thinking_round=1,
        capped=False,
        budget=reply_budget.decide(effort="low", researched=False, w_count=3),
        tone="いま向き合っている相手は大人。大人（パパ・ママ）には「〜です」「〜ます」で話す。",
    )
    first = text.splitlines()[0]
    assert first.startswith("[返事] 目標")
    assert "「〜です」「〜ます」で話す。" in first


def test_no_tone_leaves_the_reply_line_alone():
    from familiar_agent.loop import reply_budget
    from familiar_agent.loop.generator import _iter_ctx

    budget = reply_budget.decide(effort="low", researched=False, w_count=3)
    text = _iter_ctx(chain=1, max_chain=5, thinking_round=1, capped=False, budget=budget, tone="")
    # 口調が無ければ、`[返事]` の行は 出-ah-ろ の追記までのまま。
    assert text.splitlines()[0] == budget.line()
