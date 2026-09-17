"""タイマーの振る舞いの設定 3 つ（知-o・2026-09-18・`設計方針_タイマー` v0.3）。

`TIMER_SILENCE`（黙る）・`TIMER_MIC_CLOSE`（聞かない）・`TIMER_CONFIRM`（掛ける前に確かめる）。
正本は `.env`（再起動しても保持）、設定画面の「タイマー」欄で変え、保存した瞬間に効く
（`TimerTool` は呼ぶたびに読む）。`/reload` の要約にも出る。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from familiar_agent.config import AgentConfig
from familiar_agent.env_reload import LISTENER_KEYS
from familiar_agent.settings_schema import SETTINGS_FIELDS, SetupConfig, setup_config_to_env_values
from familiar_agent.tools.timer import TimerTool


def test_the_three_flags_have_the_agreed_defaults(monkeypatch):
    for k in ("TIMER_SILENCE", "TIMER_MIC_CLOSE", "TIMER_CONFIRM"):
        monkeypatch.delenv(k, raising=False)
    cfg = AgentConfig()
    assert (cfg.timer_silence, cfg.timer_mic_close, cfg.timer_confirm) == (True, True, True)


def _tool():
    return TimerTool(
        store=lambda: MagicMock(),
        oif=MagicMock(),
        speaker=lambda: "パパ",
        quiet=lambda: None,
        silence_active=lambda: False,
    )


def test_the_tool_reads_the_flags_each_time(monkeypatch):
    """設定画面で変えたら、作り直さなくても次の呼び出しから効く。"""
    tool = _tool()
    monkeypatch.setenv("TIMER_CONFIRM", "false")
    assert tool.flags().confirm is False
    monkeypatch.setenv("TIMER_CONFIRM", "true")
    assert tool.flags().confirm is True
    monkeypatch.setenv("TIMER_MIC_CLOSE", "false")
    assert tool.flags().mic_close is False


def test_the_settings_dialog_has_a_timer_section():
    keys = {f.env_key: f for f in SETTINGS_FIELDS}
    for k in ("TIMER_SILENCE", "TIMER_MIC_CLOSE", "TIMER_CONFIRM"):
        assert k in keys and keys[k].widget == "bool" and keys[k].section == "voice"
    values = setup_config_to_env_values(SetupConfig(timer_confirm=False, timer_mic_close=True))
    assert values["TIMER_CONFIRM"] == "false" and values["TIMER_MIC_CLOSE"] == "true"


def test_reload_reports_the_timer_flags():
    for k in ("TIMER_SILENCE", "TIMER_MIC_CLOSE", "TIMER_CONFIRM"):
        assert k in LISTENER_KEYS
