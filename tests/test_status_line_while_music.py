"""音楽が鳴っているあいだの状態の行は、音楽用の文で出す（出-co・2026-10-15・本人の決定ア）。

タイマー用の型 1 つで出していたので、音楽中は「聞いていない（音楽が鳴っている が鳴るまで）」と崩れ、下の行も
「止めて・一時停止・再開の言葉だけ届く。/mic on で聞く。」とタイマーの言葉だった。音楽中に届くのは、文頭に名前の
ある言葉と、音楽の言葉（操作・プレイリスト名・出-cp で名前なしでも受ける）である。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from familiar_agent.diagnostics import build_gui_diagnostics


def _snap(reason: str):
    window = MagicMock()
    window._agent_ready = True
    window._agent_running = False
    window._agent_init_failed = False
    window._agent.mic_gate_reason = lambda: reason
    window._realtime_stt.gated = False
    return build_gui_diagnostics(window)


def test_music_says_what_gets_through():
    snap = _snap("音楽が鳴っている")
    assert snap.phase == "not_listening"
    assert snap.headline == "🎵 音楽中：名前か音楽の言葉だけ届く"
    assert (
        snap.detail == "止めて・次の曲・音量・プレイリスト名は名前なしで届く。ほかは名前から話す。"
    )
    assert "mic=music" in snap.readiness and "mic=closed" not in snap.readiness


def test_a_timer_keeps_its_own_words():
    """**反証の側**：タイマーのときは、タイマーの文と印（名前が要ること・時間を聞けること・ウの 3・2026-11-13）。"""
    snap = _snap("タイマー「パスタ」")
    assert snap.headline == "🔇 聞いていない（タイマー「パスタ」 が鳴るまで）"
    assert snap.detail == (
        "名前を呼んだときの、止めて・一時停止・再開の言葉と、タイマーへの問いだけ届く。/mic on で聞く。"
    )
    assert "mic=closed" in snap.readiness and "mic=music" not in snap.readiness


def test_a_timer_named_after_music_is_still_a_timer():
    """**反証の側**：音楽かは理由の文字列が同じかで見る（`agent.MUSIC_GATE_REASON`・入口の耳と同じ見分け方）。
    タイマーの名前に「音楽」が入っても、タイマーの文で出す（ウの 3 の改め・2026-11-14・本人の承認）。"""
    snap = _snap("タイマー「音楽の練習」")
    assert snap.headline.startswith("🔇 聞いていない")
    assert "mic=closed" in snap.readiness and "mic=music" not in snap.readiness
