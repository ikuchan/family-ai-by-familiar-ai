"""P1：知覚→save の面の材料の配線（`writer_id`／`participants` を PMM から）。

観察＝エージェント自身の情景観察（書き手＝`__self__`）。会話＝話者との遣り取り
（書き手＝話者 floor DEFAULT）。`participants` は在席者で、書いた直後に `present` の面になる。

**`subject_id` は撤去した**（056・段5 の残り）。列を落としたあとも引数だけが残り、受け取って
捨てていた。誰についての記録かは `about` の面が持つ。実在の人を指す 397 件は、その全件が
その人の面を既に持っていた（2026-08-21 のダンプ）。

視点の絞りを表す `scope` は 039 で落とした。誰との遣り取りかは `actor` と `present` の面が
持っており、同じことを別の語で重ねていた。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from familiar_agent.agent import EmbodiedAgent
from familiar_agent.person_memory_manager import AGENT_SELF_ID, DEFAULT_PERSON_ID


def _mock_self(speaker_id, present):
    s = MagicMock()
    s._pmm = MagicMock()
    s._pmm.current_speaker_id = speaker_id
    s._pmm.get_present_ids.return_value = present
    return s


def test_observation_perspective_with_speaker():
    """観察はパジュが書く。誰が居たかは知覚から来る。"""
    s = _mock_self("PERSON-A", ["PERSON-A", "PERSON-B"])
    p = EmbodiedAgent._observation_perspective(s)
    assert p == dict(
        writer_id=AGENT_SELF_ID,
        participants=["PERSON-A", "PERSON-B"],
    )


def test_observation_perspective_does_not_depend_on_the_speaker():
    """話者が分からなくても観察は書ける（書き手はパジュだから）。"""
    s = _mock_self(None, [])
    p = EmbodiedAgent._observation_perspective(s)
    assert p == dict(writer_id=AGENT_SELF_ID, participants=[])


def test_conversation_perspective_with_speaker():
    s = _mock_self("PERSON-A", ["PERSON-A"])
    p = EmbodiedAgent._conversation_perspective(s)
    assert p == dict(
        writer_id="PERSON-A",
        participants=["PERSON-A"],
    )


def test_conversation_perspective_floors_to_default():
    """話者不在は floor で `default`。048 で、その `actor` は `__self__` へ寄る。"""
    s = _mock_self(None, [])
    p = EmbodiedAgent._conversation_perspective(s)
    assert p["writer_id"] == DEFAULT_PERSON_ID


def test_neither_perspective_carries_a_subject():
    """`subject_id` を渡さない（反証側）。"""
    for fn in (EmbodiedAgent._observation_perspective, EmbodiedAgent._conversation_perspective):
        assert "subject_id" not in fn(_mock_self("PERSON-A", ["PERSON-A"]))
