"""`/voice 名前` で声を登録する（知-ai 段 5・2026-10-05・本人の決定ウ→イ・`設計方針_在席と顔ぶれ` v0.1）。

名乗りは、名乗った本人の声の基準があるときにしか効かない（知-ai 段 4）。基準がまだ無い人の**最初の登録**は、
登録のためのコマンドで行う：`/voice パパ` と打ってから 10 秒以内（`VOICE_ENROLL_SEC`・仮）の声の発話を、その人の
登録の声と今日の声に足し、顔ぶれに入れて話者にする。登録の発話は会話には流さない（窓が閉じていても登録できる・
改造方針で承認した解釈）。

- 家族に無い名前は断る（`FAMILY.md` で名前に直してから人物表を引く）。
- 10 秒を過ぎたら受付を閉じ、その声は普段どおり会話へ流す。
- 声の特徴が無い入力（キーボード・短い断片）は登録に使わず、受付は開いたまま。
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import numpy as np

from familiar_agent.agent import EmbodiedAgent as Agent
from familiar_agent.config import RecognitionConfig
from familiar_agent.core.wake_window import KeyText, VoiceText
from tests.test_input_commands_before_loop_branch import _agent as _base

FAMILY = "## パパ\n- **名前**：雄輔\n- **呼び方**：パパ、ゆうすけ\n"
VOICE = np.asarray([1.0, 0.0], dtype=np.float32)


def _agent(clock):
    a = _base()
    a._family_md = FAMILY
    a.config.recognition = RecognitionConfig()
    a._pmm = MagicMock()
    a._pmm.find_person_id_by_name = MagicMock(
        side_effect=lambda n: "papa" if n in ("パパ", "雄輔") else None
    )
    store = MagicMock()
    a._voice_store = MagicMock(return_value=store)
    a._now = lambda: clock["t"]
    a._voice_enroll = None
    a._handle_voice_command = lambda ui: Agent._handle_voice_command(a, ui)
    a._enroll_voice = lambda v: Agent._enroll_voice(a, v)
    return a, store


def test_the_window_is_ten_seconds(monkeypatch):
    monkeypatch.delenv("VOICE_ENROLL_SEC", raising=False)
    assert RecognitionConfig().voice_enroll_sec == 10.0


def test_a_voice_after_the_command_is_registered_and_names_the_speaker():
    clock = {"t": 1000.0}
    a, store = _agent(clock)
    out = asyncio.run(Agent.run(a, KeyText("/voice パパ")))
    assert "パパ" in out and "10 秒" in out
    clock["t"] = 1005.0
    out = asyncio.run(Agent.run(a, VoiceText("こんにちは、パジュ", voice=VOICE)))
    assert "声を登録しました" in out
    origins = [c.args[2] for c in store.add.call_args_list]
    assert origins == ["registered", "today"]
    assert all(c.args[0] == "papa" and c.args[1] == "voice" for c in store.add.call_args_list)
    a._persons.set_active.assert_called_with("パパ")
    a._sync_pmm_speaker.assert_awaited_with("パパ")
    a._info_processing.push_utterance.assert_not_awaited()  # 会話には流さない
    assert a._voice_enroll is None  # 1 回で閉じる


def test_a_name_outside_the_family_is_refused():
    a, _ = _agent({"t": 1000.0})
    out = asyncio.run(Agent.run(a, KeyText("/voice 太郎")))
    assert "家族" in out
    assert a._voice_enroll is None


def test_after_ten_seconds_the_voice_goes_to_the_conversation():
    clock = {"t": 1000.0}
    a, store = _agent(clock)
    asyncio.run(Agent.run(a, KeyText("/voice パパ")))
    clock["t"] = 1011.0
    asyncio.run(Agent.run(a, VoiceText("こんにちは", voice=VOICE)))
    store.add.assert_not_called()
    a._info_processing.push_utterance.assert_awaited_once()
    assert a._voice_enroll is None


def test_input_without_a_voice_is_not_registered():
    clock = {"t": 1000.0}
    a, store = _agent(clock)
    asyncio.run(Agent.run(a, KeyText("/voice パパ")))
    asyncio.run(Agent.run(a, KeyText("こんにちは")))  # キーボード
    asyncio.run(Agent.run(a, VoiceText("うん")))  # 短い断片（特徴なし）
    store.add.assert_not_called()
    assert a._voice_enroll is not None  # 受付は開いたまま


def test_the_command_alone_explains_itself():
    a, _ = _agent({"t": 1000.0})
    out = asyncio.run(Agent.run(a, KeyText("/voice")))
    assert "/voice" in out
