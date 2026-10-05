"""残高切れ（クレジット切れ）を見分ける（環-z・2026-10-05・`設計方針_クレジット切れの知らせ` v0.1）。純関数。

実機 2026-09-26、軽量LLM（Gemini）が 402「Your prepayment credits are depleted」を返し続けた。残高切れは待っても
戻らないので、一時的な失敗としてやり直さない（`backends/shared._is_transient_error`）。見分けは担い手ごとの返り方を
ここに並べる。Gemini 以外は文書の上の推測で、実際に切れたときのログで見直す。

- Gemini：402、または「credits are depleted」「prepayment credits」（実物で確認）。`RESOURCE_EXHAUSTED` は回数制限の
  429 にも使われるので、それだけでは残高切れとしない。
- Anthropic：「credit balance is too low」（400 の `invalid_request_error`）。
- OpenAI 系（GLM・Kimi を含む）：`insufficient_quota`・「insufficient balance」。
- Jev（TypeSafe AI）：402。
"""

from __future__ import annotations

_MARKERS = (
    "credits are depleted",
    "prepayment credits",
    "credit balance is too low",
    "insufficient_quota",
    "insufficient balance",
    "balance is insufficient",
)


def is_credit_exhausted(err: object) -> bool:
    """`err`（例外・状態の番号・文）が残高切れを表すか。"""
    if isinstance(err, int):
        return err == 402
    code = getattr(err, "code", None) or getattr(err, "status_code", None)
    if code == 402:
        return True
    text = str(err).lower()
    return any(m in text for m in _MARKERS)


# ── 専用の情動（知らせたい）の置き場（環-z 段 2）──────────────────────────────────────
#
# 担い手（anthropic・gemini・openai・glm・kimi・jev）→ 切れたと分かった時刻（ISO）。DB（`agent_state` の
# `credit_alerts`）に置き、再起動をまたいで残す。LLM の呼び出しのたびに読むので、プロセスの中に写しを持ち、
# 変わったときだけ書く。

import contextvars  # noqa: E402
import logging  # noqa: E402
import threading  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

from . import state_json  # noqa: E402

logger = logging.getLogger(__name__)

KEY = "credit_alerts"
_lock = threading.Lock()
_cache: "dict[str, str] | None" = None
#: いまの呼び出しの中で失敗を受け止めたか（`backends/shared.watch_credit` が見る）。
failed_in_call: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "credit_failed_in_call", default=False
)


def forget_cache() -> None:
    """プロセスの写しを捨てる（次に DB から読み直す・試験と再起動の代わり）。"""
    global _cache
    with _lock:
        _cache = None


def _alerts() -> "dict[str, str]":
    global _cache
    if _cache is None:
        raw = state_json.read(KEY)
        _cache = dict(raw) if isinstance(raw, dict) else {}
    return _cache


def pending() -> "dict[str, str]":
    """いま立っている「知らせたい」（担い手 → 切れた時刻）。"""
    with _lock:
        return dict(_alerts())


def note_failure(name: str, err: object) -> bool:
    """担い手 `name` の呼び出しが失敗した。残高切れなら「知らせたい」を立て、真を返す。"""
    failed_in_call.set(True)
    if not name or not is_credit_exhausted(err):
        return False
    with _lock:
        alerts = _alerts()
        if name in alerts:
            return True
        alerts[name] = datetime.now(timezone.utc).isoformat()
        state_json.write(KEY, alerts)
    logger.warning("クレジット切れ：%s（知らせたいを立てた）", name)
    return True


def note_success(name: str) -> None:
    """担い手 `name` の呼び出しが通った。「知らせたい」が立っていれば消す（回復）。"""
    with _lock:
        alerts = _alerts()
        if name not in alerts:
            return
        del alerts[name]
        if alerts:
            state_json.write(KEY, alerts)
        else:
            state_json.clear(KEY)
    logger.info("クレジットが戻った：%s（知らせたいを消した）", name)
