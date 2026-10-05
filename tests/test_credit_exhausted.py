"""残高切れを見分け、やり直さない（環-z 段 1・2026-10-05・`設計方針_クレジット切れの知らせ` v0.1）。

実機 2026-09-26 08:27〜08:39、軽量LLM（Gemini）が 402「Your prepayment credits are depleted」を返し続けた（12 分で
240 回）。文字列に `resource_exhausted` を含むので一時的な失敗とみなされ、1 回の依頼で 3 回叩いていた。

- 見分けは 1 か所（`core/credit.is_credit_exhausted`）。担い手ごとの返り方を並べる：Gemini の 402 と「credits are
  depleted」（実物）、Anthropic の「credit balance is too low」、OpenAI 系の `insufficient_quota`、Jev の 402（どれも
  文書の上・未確認）。
- Gemini の `RESOURCE_EXHAUSTED` は回数制限の 429 にも使われるので、それだけでは残高切れとしない。
- 残高切れは一時的な失敗ではない（やり直さない）。
"""

from __future__ import annotations

import asyncio

import pytest

from familiar_agent.backends.shared import _is_transient_error, _retry_transient
from familiar_agent.core.credit import is_credit_exhausted


class _Err(Exception):
    def __init__(self, text: str, code: "int | None" = None):
        super().__init__(text)
        self.code = code


GEMINI_402 = _Err(
    "402 RESOURCE_EXHAUSTED. {'error': {'code': 402, 'message': 'Your prepayment credits are "
    "depleted. Please go to AI Studio to manage your project and billing.', 'status': "
    "'RESOURCE_EXHAUSTED'}}",
    code=402,
)
GEMINI_429 = _Err(
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, "
    "please check your plan and billing details.', 'status': 'RESOURCE_EXHAUSTED'}}",
    code=429,
)


@pytest.mark.parametrize(
    "err",
    [
        GEMINI_402,
        _Err("prepayment credits are depleted"),  # 番号が落ちても文で分かる
        _Err(
            "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
            "'message': 'Your credit balance is too low to access the Anthropic API.'}}",
            code=400,
        ),
        _Err(
            "Error code: 429 - {'error': {'message': 'You exceeded your current quota', "
            "'type': 'insufficient_quota', 'code': 'insufficient_quota'}}",
            code=429,
        ),
        _Err("jev 402", code=402),
    ],
)
def test_credit_exhaustion_is_recognised(err):
    assert is_credit_exhausted(err)


@pytest.mark.parametrize(
    "err",
    [
        GEMINI_429,  # 回数制限（待てば戻る）
        _Err("503 UNAVAILABLE high demand", code=503),
        _Err("400 invalid_argument", code=400),
        _Err("Connection reset by peer"),
    ],
)
def test_other_failures_are_not_credit_exhaustion(err):
    assert not is_credit_exhausted(err)


def test_credit_exhaustion_is_not_transient():
    assert not _is_transient_error(GEMINI_402)
    assert _is_transient_error(GEMINI_429)  # 回数制限はいままでどおりやり直す


def test_credit_exhaustion_is_not_retried():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise GEMINI_402

    with pytest.raises(_Err):
        asyncio.run(_retry_transient(fn, attempts=3, base_sec=0.0, label="t"))
    assert calls["n"] == 1
