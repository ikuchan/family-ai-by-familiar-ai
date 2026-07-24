"""P1：知覚→save の視点列配線（writer_id/subject_id/participants/scope を PMM から）。

観察＝エージェント自身の情景観察（writer=AGENT_SELF・scope=scene・subject は話者 floor DEFAULT）。
会話＝話者との遣り取り（writer=subject=話者 floor DEFAULT・scope=speaker）。participants は在席者。
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
    s = _mock_self("PERSON-A", ["PERSON-A", "PERSON-B"])
    p = EmbodiedAgent._observation_perspective(s)
    assert p == dict(
        writer_id=AGENT_SELF_ID,
        subject_id="PERSON-A",
        participants=["PERSON-A", "PERSON-B"],
        scope="scene",
    )


def test_observation_perspective_floors_to_default():
    s = _mock_self(None, [])
    p = EmbodiedAgent._observation_perspective(s)
    assert p["writer_id"] == AGENT_SELF_ID
    assert p["subject_id"] == DEFAULT_PERSON_ID  # 話者不在は floor で DEFAULT
    assert p["scope"] == "scene"


def test_conversation_perspective_with_speaker():
    s = _mock_self("PERSON-A", ["PERSON-A"])
    p = EmbodiedAgent._conversation_perspective(s)
    assert p == dict(
        writer_id="PERSON-A",
        subject_id="PERSON-A",
        participants=["PERSON-A"],
        scope="speaker",
    )


def test_conversation_perspective_floors_to_default():
    s = _mock_self(None, [])
    p = EmbodiedAgent._conversation_perspective(s)
    assert p["writer_id"] == DEFAULT_PERSON_ID
    assert p["subject_id"] == DEFAULT_PERSON_ID
    assert p["scope"] == "speaker"
