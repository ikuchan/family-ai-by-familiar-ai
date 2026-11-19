"""Tests for TTS/STT self-loop guard logic.

声が出ているか（話している）は、つないだ読み口（状態図の声の話している・環-aw 第 5 段）から読む。見張りが持つのは、
言った文の印（`remember`）と、話し終えてからの 1.2 秒（`after_speech`）と、こだまの数えだけ。
"""

from __future__ import annotations

from familiar_agent.voice_guard import VoiceLoopGuard


def test_voice_guard_blocks_self_echo_during_speaking(monkeypatch) -> None:
    ts = iter([10.0, 10.1])
    monkeypatch.setattr("familiar_agent.voice_guard._guard_now", lambda: next(ts))

    guard = VoiceLoopGuard()
    guard.attach(lambda: True)  # 状態図の声が話している
    guard.remember("[cheerful] こんにちは")

    decision = guard.check_transcript("こんにちは")

    assert decision.blocked is True
    assert decision.reason == "tts_active"
    assert decision.should_restart is False


def test_voice_guard_blocks_recent_self_echo_after_tts(monkeypatch) -> None:
    ts = iter([20.0, 20.2, 20.4, 22.0])
    monkeypatch.setattr("familiar_agent.voice_guard._guard_now", lambda: next(ts))

    guard = VoiceLoopGuard(suppression_window_secs=1.0)
    guard.remember("今日はいい天気だね")
    guard.after_speech(played=True)

    blocked = guard.check_transcript("今日はいい天気だね。")
    allowed = guard.check_transcript("窓を開ける？")

    assert blocked.blocked is True
    assert blocked.reason == "tts_suppression"
    assert allowed.blocked is False


def test_voice_guard_requests_watchdog_restart_after_repeated_echo(monkeypatch) -> None:
    ts = iter([30.0, 30.1, 30.3, 30.6, 30.9])
    monkeypatch.setattr("familiar_agent.voice_guard._guard_now", lambda: next(ts))

    guard = VoiceLoopGuard(suppression_window_secs=2.0, loop_trigger_count=3)
    guard.remember("おはよう")
    guard.after_speech(played=True)

    first = guard.check_transcript("おはよう")
    second = guard.check_transcript("おはよう")
    third = guard.check_transcript("おはよう")

    assert first.should_restart is False
    assert second.should_restart is False
    assert third.should_restart is True
    assert guard.loop_counter == 3


def test_voice_guard_reads_speaking_from_the_attached_reader() -> None:
    """話しているかは読み口から読む。つないでいなければ、出ていないと読む。"""
    guard = VoiceLoopGuard()
    assert guard.speaking is False and guard.gated is False
    voice = {"speaking": True}
    guard.attach(lambda: voice["speaking"])
    assert guard.speaking is True and guard.gated is True
    voice["speaking"] = False
    assert guard.speaking is False


def test_voice_guard_waits_only_after_a_voice_that_played(monkeypatch) -> None:
    """**反証の側**：鳴らなかった声のあとは 1.2 秒を数えない（こだまは出ていない）。"""
    monkeypatch.setattr("familiar_agent.voice_guard._guard_now", lambda: 50.0)
    guard = VoiceLoopGuard(suppression_window_secs=1.0)
    guard.remember("おやすみ")
    guard.after_speech(played=False)
    assert guard.gated is False
    assert guard.check_transcript("おやすみ").blocked is False
    guard.after_speech(played=True)
    assert guard.gated is True
