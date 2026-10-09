"""調停の試験の道具（環-ab B・2026-09-28）。

調停は、Jev が決め（分岐・深さ・動作・時期・黙る依頼・名乗り・打ち消し）、要るときだけ軽量LLM が書く
（返事・つなぎ・動作の中身・日付）。試験でも**決めることと書くことを分けて**与える。

- `jev_says(...)`：偽の Jev。問いの名前で答えを組む（`Arbiter._questions` の鍵と同じ）。
- `jev_down()`：使えない Jev（鍵が無い・時間切れ）。調停は full へ倒れる。
- `writer_says({...})`：偽の軽量LLM。書くものを辞書で返す。遅らせる・失敗させることもできる。
- `decide(jev=..., writer=..., **材料)`：`ArbiterInput` を組んで本物の `Arbiter.decide` を呼ぶ。
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.loop.arbiter import Arbiter, ArbiterInput, Decision


def _pick(key: str, confidence: float) -> dict:
    return {"choice": key, "confidence": confidence}


def _yes(flag: bool) -> dict:
    return {"noul": 1.0 if flag else 0.0}


def jev_says(
    branch: str = "light",
    *,
    effort: str = "low",
    action: str = "recall",
    refers_time: bool = False,
    quiet: "int | str | None" = None,
    lifts_quiet: bool = False,
    claimed: str = "",
    denied: str = "",
    confidence: float = 0.9,
) -> MagicMock:
    """偽の Jev。`quiet` は黙る依頼の長さ（分）か "default"（長さの指定なし）。None なら頼んでいない。"""
    answers = {
        "branch": _pick(branch, confidence),
        "effort": _pick(effort, confidence),
        "action": _pick(action, confidence),
        "refers_time": _yes(refers_time),
        "asks_quiet": _yes(quiet is not None),
        "quiet_minutes": _pick(str(quiet) if quiet is not None else "default", confidence),
        "lifts_quiet": _yes(lifts_quiet),
        "claims": _yes(bool(claimed)),
        "claimed": _pick(claimed or "other", confidence),
        "denies": _yes(bool(denied)),
        "denied": _pick(denied or "other", confidence),
    }
    # 発話の新しい問い（出-ay 段 4-4b：意味＋意味ごとの動作）にも同じ答えを写す。古い分岐で書いた試験の意図を変えずに、
    # 新しい道を通す（light→その他の会話で軽く返す・full→考えて返す・action→その道具が属する意味の道具）。
    meaning, act = _as_meaning(branch, action)
    # 黙る依頼・解く・名乗り・否定は、新しい問いでは意味と動作になる（段 4-4c）。
    if quiet is not None:
        meaning, act = "time", "quiet"
    elif lifts_quiet:
        meaning, act = "time", "lift_quiet"
    elif claimed:
        meaning, act = "claim", claimed
    elif denied:
        meaning, act = "deny", denied
    answers["meaning"] = _pick(meaning, confidence)
    answers[f"action_{meaning}"] = _pick(act, confidence)
    jev = MagicMock()
    jev.available = True
    jev.ask = AsyncMock(return_value=JevAnswer(ok=True, answers=answers))
    return jev


_MUSIC = {"play_music", "stop_music", "next_track", "music_volume"}
_TIME = {
    "set_timer",
    "cancel_timer",
    "pause_timer",
    "resume_timer",
    "set_alarm",
    "cancel_alarm",
    "start_stopwatch",
    "stop_stopwatch",
}


def _as_meaning(branch: str, action: str) -> "tuple[str, str]":
    """古い分岐の答え → 発話の新しい問いの（意味, 動作）。"""
    if branch == "light":
        return "other", "reply_light"
    if branch != "action":
        return "other", "reply_full"
    if action in _MUSIC:
        return "music", action
    if action in _TIME:
        return "time", action
    if action in ("look", "see"):
        return "look", action
    if action in ("confirm", "decline"):
        return "confirm", action
    return "research", action


def jev_in_order(*jevs: MagicMock) -> MagicMock:
    """反復ごとに違う答えを返す偽の Jev（`jev_says(...)` を順に並べる）。使い切ったら最後の答えを繰り返す。"""
    answers = [j.ask.return_value for j in jevs]

    async def ask(*_a, **_k):
        return answers.pop(0) if len(answers) > 1 else answers[0]

    jev = MagicMock()
    jev.available = True
    jev.ask = AsyncMock(side_effect=ask)
    return jev


def jev_down() -> MagicMock:
    """使えない Jev（鍵が無い・時間切れ・形の誤り）。"""
    jev = MagicMock()
    jev.available = False
    jev.ask = AsyncMock(return_value=JevAnswer(ok=False, answers={}, error="down"))
    return jev


def writer_says(
    texts: "dict | str | None" = None,
    *,
    delay: float = 0.0,
    raises: "BaseException | None" = None,
) -> MagicMock:
    """偽の軽量LLM。`texts` は書くもの（辞書なら JSON にして返す・文字列ならそのまま）。"""
    reply = json.dumps(texts or {}, ensure_ascii=False) if not isinstance(texts, str) else texts

    async def complete(prompt, max_tokens=300, *, system=None):
        if delay:
            await asyncio.sleep(delay)
        if raises is not None:
            raise raises
        return reply

    w = MagicMock()
    w.complete = AsyncMock(side_effect=complete)
    return w


def writer_in_order(*texts: dict) -> MagicMock:
    """呼ばれるたびに違うものを書く偽の軽量LLM。使い切ったら最後を繰り返す。"""
    replies = [json.dumps(t, ensure_ascii=False) for t in texts]

    async def complete(prompt, max_tokens=300, *, system=None):
        return replies.pop(0) if len(replies) > 1 else replies[0]

    w = MagicMock()
    w.complete = AsyncMock(side_effect=complete)
    return w


def prompt_of(writer: MagicMock) -> str:
    """軽量LLM に実際に渡ったプロンプト。"""
    return writer.complete.await_args.args[0]


def system_of(writer: MagicMock) -> str:
    """軽量LLM に実際に渡ったシステム文。"""
    return writer.complete.await_args.kwargs.get("system") or ""


async def decide(
    *, jev, writer=None, timeout: "float | None" = 2.0, min_conf: float = 0.6, **inp
) -> Decision:
    """材料（`ArbiterInput` の欄）から本物の `Arbiter.decide` を呼ぶ。"""
    inp.setdefault("workspace_ctx", "")
    arbiter = Arbiter(jev=jev, writer=writer or writer_says(), min_conf=min_conf, timeout=timeout)
    return await arbiter.decide(ArbiterInput(**inp))
