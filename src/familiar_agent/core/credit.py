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
