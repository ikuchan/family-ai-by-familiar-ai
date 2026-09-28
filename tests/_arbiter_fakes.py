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
    jev = MagicMock()
    jev.available = True
    jev.ask = AsyncMock(return_value=JevAnswer(ok=True, answers=answers))
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
