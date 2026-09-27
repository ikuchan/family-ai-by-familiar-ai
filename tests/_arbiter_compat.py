"""試験用：古い調停の呼び方（`arbitrate(backend, …)`・`_parse(文字列)`）で、新しい `Arbiter` を通す（出-au 段 5-7d）。

古い調停は、判定（分岐・深さ・動作・黙る依頼・名乗り・打ち消し・時期）と文章（返事・つなぎ・検索語・道具の入力・日付）を
1 つの JSON で返していた。守りや分岐を確かめる試験の多くは、その JSON を偽の軽量LLM に返させている。ここでは同じ JSON を
**判定の部分は偽の Jev の答えに**振り分け、**文章の部分はその偽の軽量LLM が書く**（`Arbiter` の文章の口が要るものだけを拾う）。

偽の軽量LLM が JSON を返さない（例外・時間切れ・読めない返事）ときは、判定は light として通し、文章の口で失敗させる——
いままでと同じく full へ倒れる。
"""

from __future__ import annotations

import json
import re
from dataclasses import fields
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.loop.arbiter import Arbiter, ArbiterInput, assemble

_INPUT_FIELDS = {f.name for f in fields(ArbiterInput)}


def _peek(backend) -> "dict | None":
    """偽の軽量LLM が返すはずの JSON（`complete` の return_value）。読めなければ None。"""
    rv = getattr(getattr(backend, "complete", None), "return_value", None)
    if not isinstance(rv, str):
        return None
    m = re.search(r"\{.*\}", rv, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def _c(pick) -> dict:
    return {"choice": pick, "confidence": 0.9}


def jev_from(data: "dict | None"):
    """古い JSON の判定の部分を、偽の Jev の答えにする。"""
    if data is None:
        answers = {"branch": _c("light")}  # 文章の口で失敗させて full へ倒す
    else:
        answers = {"branch": _c(str(data.get("branch", "")).strip().lower())}
        answers["effort"] = _c(str(data.get("effort", "low")))
        answers["action"] = _c(str(data.get("action", "") or "recall"))
        answers["refers_time"] = {"noul": 1.0 if data.get("time_ref") else 0.0}
        minutes = (
            int(data.get("silence_minutes", 0) or 0)
            if str(data.get("silence_minutes", 0)).lstrip("-").isdigit()
            else 0
        )
        answers["asks_quiet"] = {"noul": 1.0 if minutes else 0.0}
        answers["quiet_minutes"] = _c("default" if minutes < 0 else str(minutes))
        answers["lifts_quiet"] = {"noul": 1.0 if data.get("lift_silence") else 0.0}
        claim = str(data.get("speaker_claim", "") or "")
        answers["claims"] = {"noul": 1.0 if claim else 0.0}
        answers["claimed"] = _c(claim or "other")
        denied = str(data.get("not_person", "") or "")
        answers["denies"] = {"noul": 1.0 if denied else 0.0}
        answers["denied"] = _c(denied or "other")
    jev = MagicMock()
    jev.available = True
    jev.ask = AsyncMock(return_value=JevAnswer(ok=True, answers=answers))
    return jev


async def arbitrate(backend, **kw):
    """古い `arbitrate` の呼び方で `Arbiter.decide` を通す。"""
    data = _peek(backend)
    inp = ArbiterInput(**{k: v for k, v in kw.items() if k in _INPUT_FIELDS})
    if data is not None and data.get("speaker_claim"):
        # 名乗りは家族の呼び方の選択肢から選ぶ。試験の名前をそのまま選べるよう、家族の記述に足す。
        name = str(data["speaker_claim"])
        if name not in (inp.family_md or ""):
            inp.family_md = (
                inp.family_md or ""
            ) + f"\n## {name}\n- 名前: {name}\n- 呼び方: {name}\n"
    return await Arbiter(
        jev=jev_from(data), writer=backend, min_conf=0.6, timeout=kw.get("timeout")
    ).decide(inp)


def _parse(reply: str, **kw):
    """古い `_parse(文字列)` の呼び方で `assemble` を通す。読めない文字列は None。"""
    m = re.search(r"\{.*\}", reply or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None
    return assemble(data, **kw) if isinstance(data, dict) else None


_REAL_DECIDE = Arbiter.decide


async def decide_through_the_writer(self, inp):
    """ループの試験用：`Arbiter.decide` の差し替え。

    ループの試験の多くは、古い調停の JSON を `agent._utility_backend`（文章の口）に返させている。その口を
    **1 回だけ**呼んで（プロンプトは Jev に送る文・並べた返事〈`side_effect`〉を 1 つだけ使う）、判定を偽の Jev の
    答えに、文章をその同じ返事にして、本物の `decide` を通す。読めない返事・例外・時間切れは full へ倒れる。
    """
    import asyncio

    try:
        reply = await asyncio.wait_for(
            self._writer.complete(self._state(inp), 300, system=None),
            timeout=self._timeout if self._timeout is not None else 5.0,
        )
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001  時間切れ・失敗は full へ
        reply = None
    data = _peek_text(reply)
    writer = MagicMock()
    writer.complete = AsyncMock(return_value=reply if isinstance(reply, str) else "")
    fake = Arbiter(jev=jev_from(data), writer=writer, min_conf=self._min_conf, timeout=5.0)
    return await _REAL_DECIDE(fake, inp)


def _peek_text(reply) -> "dict | None":
    holder = MagicMock()
    holder.complete = MagicMock(return_value=reply)
    return _peek(holder)
