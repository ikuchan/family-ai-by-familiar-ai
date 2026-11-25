"""先回りして調べたことの知らせ待ち（記-ag 段 2・段 3・2026-11-25・`設計方針_先回りして調べておく`）。

段 2（`keep`）：SEEKING の求めで調べものの結果が届いて閉じたら、軽量LLM に、家族に知らせる値打ちがあるかと、知らせる
中身と、何時間で古くなるかを聞く。値打ちがあれば知らせ待ち（`agent_state` の `STATE_KEY`）に置く。
段 3（`take_due`・`content`）：T（自律機構）の輪が、そばに人が居て黙っていなければ、古くなっていず知らせていないものを
取り出して知らせ済みにし、`機器` の求め `KIND` を積む。話すかどうかと言い方は主LLM が決め、在席の門はいつものまま。門に
止められたら、ほかの言えなかったことと同じく「言いたかったこと」として記録に残り、差し出し直しはしない。古くなったものは
知らせずに捨てる。知らせ済みも古くなるまで残し、判定に見せて同じことを何度も知らせない（同じかは字の一致だけで比べ、
言い換えは判定の頼みの言葉で防ぐ）。

知らせ待ちはパジュの状態（耳・頭・口・体が何をしているか）ではなく、調べた結果の記録なので、状態図には描かない。
DB の読み書きは同期なので、呼び手が `asyncio.to_thread` で回す（`keep` は中で回す）。読んで書き直すところは `_LOCK` で
順に回す（判定の裏の仕事と T の輪が重なっても、片方の書き込みを消さない）。読めなかったときは書かない（空として上書き
しない）。値は `config.DriveConfig` の `prefetch_*`（どれも仮）。

出-as（言えなかったことを保留して人が来たら配ることはしない）の例外で、先回りして調べた結果に限る（本人の決定）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import re
from datetime import datetime, timedelta

from ..config import DriveConfig
from ..core import state_json
from ..core.helpers import strip_code_fence

logger = logging.getLogger(__name__)

STATE_KEY = "prefetched_notices"
KIND = "調べておいたこと"  # `機器` の求めの種類
TIMEOUT_SEC = 60.0
PER_TURN = 3  # 1 つの求めで知らせる数（求めの言葉は 500 字で切られる）
_LOCK = threading.Lock()  # 知らせ待ちを読んで書き直すところを順に回す
MAX_TOKENS = 300
_OBJECT = re.compile(r"\{.*\}", re.S)

PROMPT = """あなたは家庭のロボット。家族に聞かれる前に、自分から次のことを調べた。

[調べた結果]
{results}

[もう知らせ待ちにあるか、知らせたこと]
{known}

家族に知らせる値打ちがあるかを決める。値打ちがあるのは、家族が気にかけていそうなことで、新しい結果や情報が分かった
とき（試合の結果、天気の変わり目など）。調べても分からなかった、前から知られていること、家族に関係が薄いことは
値打ちが無い。もう知らせ待ちにあるか知らせたことと同じなら、値打ちは無い。

出力は JSON だけにする（説明は書かない）：
{{"tell": true か false, "what": "知らせる中身（1〜2 文・事実だけ）", "hours": 何時間で古くなるか（数）}}
"""


def _parse(raw: str) -> "tuple[str, float] | None":
    """返りを（知らせる中身, 古くなるまでの時間）にする。値打ちが無い・読めなければ None。"""
    m = _OBJECT.search(strip_code_fence(str(raw or "")))
    if m is None:
        return None
    try:
        got = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(got, dict) or got.get("tell") is not True:
        return None
    what = str(got.get("what") or "").strip()
    if not what:
        return None
    hours = got.get("hours")
    cfg = DriveConfig()
    if (
        isinstance(hours, bool)
        or not isinstance(hours, (int, float))
        or not math.isfinite(hours)
        or hours <= 0
    ):
        hours = cfg.prefetch_default_hours
    return what, min(float(hours), cfg.prefetch_max_hours)


def _items() -> "list[dict] | None":
    """知らせ待ちの項目。無ければ空、読めなければ None（呼び手は書かない）。"""
    ok, raw = state_json.read_checked(STATE_KEY)
    if not ok:
        return None
    return [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []


def _time(x: dict, key: str) -> "datetime | None":
    """項目の時刻。無い・読めない・時差が無いときは None（読めない項目で輪を止めない）。"""
    try:
        t = datetime.fromisoformat(str(x.get(key)))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo is not None else None


def _fresh(items: "list[dict]", now: datetime) -> "list[dict]":
    out = []
    for x in items:
        until = _time(x, "until")
        if until is not None and until > now and x.get("what"):
            out.append(x)
    return out


async def keep(agent, results: str, *, now: datetime) -> bool:
    """調べた結果に知らせる値打ちがあれば、知らせ待ちに置く。置いたら True。

    もう知らせ待ちにあるもの、知らせたもの（古くなっていないもの）を判定に見せ、同じことは置かない。軽量LLM が無ければ
    聞かない（主LLM で毎回聞かない）。呼べない・読めないときは置かずに続ける。
    """
    backend = agent._utility_backend
    if backend is agent.backend or not results.strip():
        return False
    items = await asyncio.to_thread(_items)
    if items is None:
        return False
    known = _fresh(items, now)
    prompt = PROMPT.format(
        results=results[:2000],
        known="\n".join(f"- {x['what']}" for x in known) or "（無い）",
    )
    logger.debug("先回りの知らせの判定の頼み（%d 字）：%s", len(prompt), prompt)
    try:
        raw = await asyncio.wait_for(backend.complete(prompt, max_tokens=MAX_TOKENS), TIMEOUT_SEC)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        logger.warning("先回りの知らせの判定を呼べなかった（置かない）", exc_info=True)
        return False
    got = _parse(raw)
    if got is None or any(x["what"] == got[0] for x in known):
        logger.info("先回りして調べた結果：知らせる値打ちは無い")
        return False
    what, hours = got
    item = {
        "what": what,
        "at": now.isoformat(),
        "until": (now + timedelta(hours=hours)).isoformat(),
    }
    if not await asyncio.to_thread(_append, item, now, DriveConfig().prefetch_max_items):
        return False
    logger.info(
        "先回りして調べた結果を知らせ待ちに置いた（%d 字・%.0f 時間で古くなる）", len(what), hours
    )
    return True


def _append(item: dict, now: datetime, keep_n: int) -> bool:
    """知らせ待ちに 1 件足す（古いものは捨て、多ければ古いほうから捨てる）。読めなければ書かない。"""
    with _LOCK:
        latest = _items()
        if latest is None:
            return False
        return state_json.write(STATE_KEY, (_fresh(latest, now) + [item])[-keep_n:])


def take_due(*, now: datetime) -> "list[str]":
    """知らせ待ちのうち、古くなっていず知らせていないものを `PER_TURN` 件まで取り出し、知らせ済みにする。

    古くなったものは捨てる。書けなければ何も返さない（書けないまま毎回知らせない）。
    """
    with _LOCK:
        items = _items()
        if not items:
            return []
        fresh = _fresh(items, now)
        due: "list[str]" = []
        for x in fresh:
            if x.get("told"):
                continue
            if len(due) >= PER_TURN:
                break
            x["told"] = True
            due.append(str(x["what"]))
        if not due and len(fresh) == len(items):
            return []
        if not state_json.write(STATE_KEY, fresh):
            return []
    if len(fresh) < len(items):
        logger.info("先回りの知らせ：古くなった %d 件を知らせずに捨てた", len(items) - len(fresh))
    return due


def is_turn(request_text: str) -> bool:
    """その求めが、先回りの知らせの `機器` の求めか（求めの言葉は「[種類] 中身」）。"""
    return str(request_text or "").startswith(f"[{KIND}]")


def content(whats: "list[str]") -> str:
    """`機器` の求めの中身。主LLM が、そばに居る家族に知らせるかと言い方を決める。"""
    return (
        "家族に聞かれる前に、先回りして調べておいたこと。そばに居る家族に、短く知らせる：\n"
        + "\n".join(f"- {w}" for w in whats)
    )
