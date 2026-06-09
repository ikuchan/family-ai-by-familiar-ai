"""Tests for PMM on_switch → DesireSystem.update_active_companion wiring (Issue #2 Item5)."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock


class TestUpdateActiveCompanion:
    """DesireSystem.update_active_companion() updates companion-related drive prompts."""

    def _make_desires(self, name: str = "Kouta"):
        from familiar_agent.desires import DesireSystem
        return DesireSystem(companion_name=name)

    def test_updates_companion_name(self):
        d = self._make_desires("Kouta")
        d.update_active_companion("Hana")
        assert d._companion_name == "Hana"

    def test_no_op_when_same_name(self):
        d = self._make_desires("Kouta")
        original_spec = d._drive_specs["greet_companion"]
        d.update_active_companion("Kouta")
        assert d._drive_specs["greet_companion"] is original_spec

    def test_no_op_when_empty_name(self):
        d = self._make_desires("Kouta")
        d.update_active_companion("")
        assert d._companion_name == "Kouta"

    def test_rebuilds_greet_companion_prompt(self):
        d = self._make_desires("Kouta")
        d.update_active_companion("Hana")
        assert "Hana" in d._drive_specs["greet_companion"].prompt_text

    def test_rebuilds_worry_companion_prompt(self):
        d = self._make_desires("Kouta")
        d.update_active_companion("Hana")
        assert "Hana" in d._drive_specs["worry_companion"].prompt_text

    def test_rebuilds_share_memory_prompt(self):
        d = self._make_desires("Kouta")
        d.update_active_companion("Hana")
        assert "Hana" in d._drive_specs["share_memory"].prompt_text

    def test_rebuilds_attachment_prompt(self):
        d = self._make_desires("Kouta")
        d.update_active_companion("Hana")
        assert "Hana" in d._drive_specs["attachment"].prompt_text

    def test_rebuilds_care_prompt(self):
        d = self._make_desires("Kouta")
        d.update_active_companion("Hana")
        assert "Hana" in d._drive_specs["care"].prompt_text

    def test_preserves_drive_levels(self):
        d = self._make_desires("Kouta")
        d.boost("greet_companion", 0.5)
        level_before = d._desires["greet_companion"]
        d.update_active_companion("Hana")
        assert d._desires["greet_companion"] == level_before


class TestOnPmmSpeakerSwitch:
    """EmbodiedAgent._on_pmm_speaker_switch() updates desires via PMM callback."""

    def _make_agent_with_desires(self):
        from familiar_agent.agent import EmbodiedAgent
        from familiar_agent.desires import DesireSystem

        agent = EmbodiedAgent.__new__(EmbodiedAgent)

        desires = DesireSystem(companion_name="Kouta")
        agent._desires_ref = desires

        mock_pmm = MagicMock()
        mock_pmm.get_speaker_info = MagicMock(
            return_value={"display_name": "Hana", "name": "hana"}
        )
        agent._pmm = mock_pmm

        return agent, desires

    @pytest.mark.asyncio
    async def test_callback_updates_companion_name(self):
        agent, desires = self._make_agent_with_desires()
        await agent._on_pmm_speaker_switch(None, "some-uuid")
        assert desires._companion_name == "Hana"

    @pytest.mark.asyncio
    async def test_callback_no_op_when_no_desires_ref(self):
        from familiar_agent.agent import EmbodiedAgent
        agent = EmbodiedAgent.__new__(EmbodiedAgent)
        agent._desires_ref = None
        mock_pmm = MagicMock()
        agent._pmm = mock_pmm
        # Should not raise
        await agent._on_pmm_speaker_switch(None, "some-uuid")
        mock_pmm.get_speaker_info.assert_not_called()

    @pytest.mark.asyncio
    async def test_callback_no_op_when_no_display_name(self):
        agent, desires = self._make_agent_with_desires()
        agent._pmm.get_speaker_info = MagicMock(return_value={"display_name": "", "name": ""})
        await agent._on_pmm_speaker_switch(None, "some-uuid")
        assert desires._companion_name == "Kouta"  # unchanged
