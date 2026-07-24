"""Gemini の一時的エラー（503/429）を指数バックオフでリトライする純ロジック。

ダッシュボードで RPM/TPM に余裕があり、ログは 503 UNAVAILABLE（server 過負荷）＝一時的。
恒久エラー（400 等）は即諦める。最終失敗は呼び出し側が空文字へ落とす（挙動は従来どおり）。
"""

from __future__ import annotations

import asyncio

import pytest

from familiar_agent.backend import _is_transient_error, _retry_transient


# ── 一時的か恒久かの判定 ─────────────────────────────────────────────────────

def test_transient_true_for_503_and_429():
    assert _is_transient_error(RuntimeError("503 UNAVAILABLE. high demand")) is True
    assert _is_transient_error(RuntimeError("429 RESOURCE_EXHAUSTED")) is True


def test_transient_false_for_permanent():
    assert _is_transient_error(RuntimeError("400 INVALID_ARGUMENT")) is False
    assert _is_transient_error(ValueError("bad schema")) is False


# ── リトライ挙動（base_sec=0 で待たない） ────────────────────────────────────

def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("503 UNAVAILABLE")
        return "ok"

    out = asyncio.run(_retry_transient(fn, attempts=3, base_sec=0.0, label="t"))
    assert out == "ok"
    assert calls["n"] == 3


def test_retry_reraises_permanent_immediately():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise RuntimeError("400 INVALID_ARGUMENT")

    with pytest.raises(RuntimeError):
        asyncio.run(_retry_transient(fn, attempts=3, base_sec=0.0, label="t"))
    assert calls["n"] == 1  # 恒久エラーはリトライしない


def test_retry_exhausts_on_persistent_transient():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise RuntimeError("503 UNAVAILABLE")

    with pytest.raises(RuntimeError):
        asyncio.run(_retry_transient(fn, attempts=3, base_sec=0.0, label="t"))
    assert calls["n"] == 3  # attempts 回で打ち切り
