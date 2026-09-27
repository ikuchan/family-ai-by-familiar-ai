"""Jev（判断専用モデル）に決めさせる判定（出-au 段 5・`設計方針_判定の段` §2.2）。

判定ごとに「送る文を組む・質問を組む・答えを読む・倒し先へ倒す」をここに置く。**Jev が使えないとき**（鍵が無い・
失敗・時間切れ）と、**確信度がしきい値（`jev_confidence_min`・0.6〔仮〕）より低いとき**は、判定ごとの倒し先へ倒す。
倒し先は「迷ったときに害が小さい側」（§2.2.1）。判定はここでは例外を投げない。
"""

from __future__ import annotations

import logging

from ..backends.jev import choice

logger = logging.getLogger(__name__)

#: 考え直すか（§2.2.3・§2.4）。
RETHINK = "考え直す"
AS_IS = "そのまま出す"


def picked(answer, key: str, min_conf: float) -> "str | None":
    """Choice の答えを読む。失敗・答えが無い・確信度が `min_conf` 未満なら None（倒し先へ）。"""
    if not getattr(answer, "ok", False):
        return None
    got = (getattr(answer, "answers", None) or {}).get(key) or {}
    if float(got.get("confidence", 0.0) or 0.0) < min_conf:
        return None
    pick = got.get("choice")
    return str(pick) if pick else None


async def _ask(client, state: str, questions: dict):
    """Jev に聞く。使えない・例外は None。"""
    if client is None or not getattr(client, "available", False):
        return None
    try:
        return await client.ask(state, questions)
    except Exception as e:  # noqa: BLE001
        logger.warning("Jev の判定に失敗した：%s", type(e).__name__)
        return None


async def judge_rethink(
    client, *, question: str, draft: str, added: "list[str]", min_conf: float
) -> str:
    """主LLM が考えているあいだに言い足された言葉で、返事を考え直すべきか（§2.4）。倒し先は「そのまま出す」。"""
    lines = "\n".join(f"- 「{t}」" for t in added)
    state = (
        f"[人が最初に言ったこと]\n{question}\n\n"
        f"[家のロボット（パジュ）が書いた返事の下書き]\n{draft}\n\n"
        f"[下書きを考えているあいだに、人が言い足したこと]\n{lines}"
    )
    questions = {
        "rethink": choice(
            "言い足されたことを踏まえて、返事の下書きを書き直すべきか",
            {
                "rethink": "書き直すべき。言い足されたことが問いを変える・絞る・正すので、下書きのままでは合わない",
                "as_is": "そのまま出してよい。言い足されたことは別の話か、下書きの答えを変えない",
            },
        )
    }
    pick = picked(await _ask(client, state, questions), "rethink", min_conf)
    verdict = RETHINK if pick == "rethink" else AS_IS
    logger.info("Jev 判定 考え直すか：%s（言い足し %d 件）", verdict, len(added))
    return verdict
