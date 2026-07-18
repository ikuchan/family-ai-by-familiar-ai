"""Tests for _evaluate_emotion_pad（評価器の PAD 出力・A_gate・解析・W2b-2）。

評価器（軽量LLM）が P/Pn/Dom を出す。A は機械 arousal。A<A_gate は評価器を呼ばず
P/Pn/Dom＝M（mood）。解析失敗・例外は mood フォールバック。値は [0,1] クランプ。
"""

from __future__ import annotations

import asyncio

from familiar_agent.agent import _evaluate_emotion_pad, A_GATE
from familiar_agent.mood_register import MoodPAD


class _FakeBackend:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls = 0

    async def complete(self, prompt: str, max_tokens: int) -> str:
        self.calls += 1
        return self._reply


_MOOD = MoodPAD(0.6, 0.3, 0.5, 0.55)


def _run(coro):
    return asyncio.run(coro)


# ── 1. A<A_gate：評価器を呼ばず mood ベース ─────────────────────────────────
def test_below_gate_uses_mood_without_backend() -> None:
    backend = _FakeBackend("0.9 0.1 0.9")
    pad = _run(_evaluate_emotion_pad(backend, "text", _MOOD, arousal=A_GATE - 0.05))
    assert backend.calls == 0
    assert pad == MoodPAD(_MOOD.p, _MOOD.pn, A_GATE - 0.05, _MOOD.dom)


# ── 2. A>=A_gate：3数値を解析して PAD（A は機械値） ─────────────────────────
def test_at_gate_parses_three_numbers() -> None:
    backend = _FakeBackend("0.7 0.2 0.6")
    pad = _run(_evaluate_emotion_pad(backend, "text", _MOOD, arousal=0.8))
    assert backend.calls == 1
    assert pad == MoodPAD(0.7, 0.2, 0.8, 0.6)


# ── 3. 解析失敗 → mood フォールバック ───────────────────────────────────────
def test_parse_failure_falls_back_to_mood() -> None:
    backend = _FakeBackend("I think it's happy!")
    pad = _run(_evaluate_emotion_pad(backend, "text", _MOOD, arousal=0.8))
    assert pad == MoodPAD(_MOOD.p, _MOOD.pn, 0.8, _MOOD.dom)


# ── 4. 範囲外はクランプ ─────────────────────────────────────────────────────
def test_out_of_range_values_are_clamped() -> None:
    backend = _FakeBackend("1.5 -0.1 0.6")
    pad = _run(_evaluate_emotion_pad(backend, "text", _MOOD, arousal=0.8))
    assert pad == MoodPAD(1.0, 0.0, 0.8, 0.6)
