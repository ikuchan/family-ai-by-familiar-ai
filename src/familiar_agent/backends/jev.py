"""Jev（判断専用モデル・TypeSafe AI）を呼ぶ口（出-au 段 4-1・`設計方針_判定の段` §2.2）。

Jev は文章を書かず、判断だけを型で返す。`state`（文）と `questions`（名前をつけた質問の組）を送ると、質問ごとに
**Choice**（選んだもの・全選択肢の確率・確信度）／**Score**（段の確率で重みづけた値）／**Noul**（はいの確率）で返る
（`POST https://api.typesafe.ai/v1/systemone`・`docs.typesafe.ai` の quickstart と API の頁）。

判定の段では**失敗しても例外にしない**。鍵が無い・時間切れ・429（回数の上限）・529（混雑）・形の誤りは、
`JevAnswer(ok=False, error=…)` を返し、呼び手が判定ごとの既定へ倒す。判定は待たせる時間そのものなので、ここでは
やり直さない（やり直すかは呼び手が決める）。HTTP は入っている `aiohttp` で呼ぶ（公式の SDK は入れない）。
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..core import credit
from ..core.aio import wait_within

logger = logging.getLogger(__name__)

URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"

#: `post(url, headers, payload, timeout) -> (status, body)`。試験では偽物を渡す。
Post = Callable[[str, dict, dict, float], Awaitable["tuple[int, Any]"]]


def choice(instructions: str, criteria: "dict[str, str]") -> dict:
    """選択肢から 1 つ選ぶ質問（選択肢は 255 個まで・キー → 説明）。"""
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def score(instructions: str, levels: "list[str]") -> dict:
    """順序のある段（2〜10 段）で測る質問。"""
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def noul(instructions: str) -> dict:
    """はい／いいえの質問。はいの確率（0〜1）が返る。"""
    return {"type": "noul", "instructions": instructions}


@dataclass(frozen=True)
class JevAnswer:
    """Jev の返り。`answers` は質問の名前 → 答え（型ごとの欄をそのまま持つ）。"""

    ok: bool
    answers: "dict[str, dict]" = field(default_factory=dict)
    model: str = ""
    seconds: float = 0.0
    error: str = ""


async def _aiohttp_post(
    url: str, headers: dict, payload: dict, timeout: float
) -> "tuple[int, Any]":
    import aiohttp

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            try:
                body = await resp.json(content_type=None)
            except Exception:  # noqa: BLE001
                body = None
            return resp.status, body


class JevClient:
    """Jev を呼ぶ口。鍵が無ければ呼ばない（`available`）。"""

    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = DEFAULT_MODEL,
        timeout: float = 10.0,
        post: "Post | None" = None,
    ) -> None:
        self._api_key = api_key or ""
        self.model = model or DEFAULT_MODEL
        self.timeout = float(timeout)
        self._post: Post = post or _aiohttp_post

    @classmethod
    def from_env(cls, *, timeout: float = 10.0) -> "JevClient":
        """鍵は `JEV_API_KEY`、モデルは `JEV_MODEL`（既定 `jev-latest`）。"""
        return cls(
            api_key=os.environ.get("JEV_API_KEY", ""),
            model=os.environ.get("JEV_MODEL", "") or DEFAULT_MODEL,
            timeout=timeout,
        )

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    async def ask(self, state: "str | dict | list", questions: "dict[str, dict]") -> JevAnswer:
        """質問の組を送り、答えを返す。失敗は例外にせず `ok=False` で返す。"""
        if not self.available:
            return JevAnswer(ok=False, error="鍵が無い")
        payload = {"state": state, "model": self.model, "questions": questions}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        started = time.monotonic()
        try:
            # `wait_for` はキャンセルを握りつぶしうる（3.11・環-aa）。判定は駆動体の上で待つので `wait_within`。
            status, body = await wait_within(
                self._post(URL, headers, payload, self.timeout), self.timeout
            )
        except asyncio.TimeoutError:
            return self._failed(started, f"時間切れ（{self.timeout:.1f} 秒）")
        except Exception as e:  # noqa: BLE001
            return self._failed(started, f"{type(e).__name__}")
        if status != 200:
            credit.note_failure("jev", status)  # 402 なら残高切れの「知らせたい」を立てる（環-z）
            return self._failed(started, f"HTTP {status}")
        answers = body.get("answers") if isinstance(body, dict) else None
        if not isinstance(answers, dict):
            return self._failed(started, "返りの形が違う")
        seconds = time.monotonic() - started
        credit.note_success("jev")  # 通った＝残高は戻っている（環-z）
        logger.info("Jev %d 問 %.2f 秒", len(questions), seconds)
        return JevAnswer(
            ok=True, answers=answers, model=str(body.get("model", "")), seconds=seconds
        )

    @staticmethod
    def _failed(started: float, error: str) -> JevAnswer:
        logger.warning("Jev に聞けなかった：%s", error)
        return JevAnswer(ok=False, seconds=time.monotonic() - started, error=error)
