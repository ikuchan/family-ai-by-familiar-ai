"""残高切れで「知らせたい」が立ち、通れば消える（環-z 段 2・2026-10-05・`設計方針_クレジット切れの知らせ` v0.1）。

残高切れを見分けたら、担い手ごとに**専用の情動**（知らせたい）が立つ。置き場は DB（`agent_state` の鍵
`credit_alerts`・担い手 → 切れた時刻）で、再起動をまたいで残る。その担い手の呼び出しが一度でも通ったら消える。

担い手の口（`stream_turn`・`complete`・`complete_with_image`）は `backends/shared.watch_credit` で包む。例外が外へ
出れば立て、出なければ（中で失敗を受け止めていなければ）消す。失敗を受け止めて空の返事を返す口は、受け止める箇所で
`core/credit.note_failure` を呼ぶ（そうしないと外からは失敗が見えない）。Jev は返りの状態の番号で見る。
"""

from __future__ import annotations

import asyncio

import pytest

from familiar_agent.core import credit

CREDIT = type("E", (Exception,), {})("402 RESOURCE_EXHAUSTED prepayment credits are depleted")
OTHER = RuntimeError("503 UNAVAILABLE")


@pytest.fixture
def db(monkeypatch):
    stored: dict = {"v": None, "writes": 0}

    def read(key):
        assert key == credit.KEY
        return stored["v"]

    def write(key, value):
        stored["v"] = value
        stored["writes"] += 1
        return True

    def clear(key):
        stored["v"] = None
        stored["writes"] += 1
        return True

    monkeypatch.setattr(credit.state_json, "read", read)
    monkeypatch.setattr(credit.state_json, "write", write)
    monkeypatch.setattr(credit.state_json, "clear", clear)
    credit.forget_cache()
    yield stored
    credit.forget_cache()


def test_a_credit_failure_raises_the_alert_and_success_clears_it(db):
    assert credit.note_failure("gemini", CREDIT) is True
    assert set(credit.pending()) == {"gemini"}
    credit.note_success("gemini")
    assert credit.pending() == {}


def test_other_failures_raise_nothing(db):
    assert credit.note_failure("gemini", OTHER) is False
    assert credit.pending() == {}


def test_the_db_is_written_only_when_it_changes(db):
    credit.note_failure("gemini", CREDIT)
    credit.note_failure("gemini", CREDIT)
    credit.note_success("anthropic")  # 立っていない担い手
    assert db["writes"] == 1
    credit.note_success("gemini")
    credit.note_success("gemini")
    assert db["writes"] == 2


def test_the_alert_survives_a_restart(db):
    credit.note_failure("anthropic", CREDIT)
    credit.forget_cache()  # 起動し直した
    assert set(credit.pending()) == {"anthropic"}


# ── 担い手の口を包む ─────────────────────────────────────────────────────────


class _Backend:
    credit_name = "gemini"

    def __init__(self, behaviour):
        self.behaviour = behaviour

    async def _run(self):
        if self.behaviour == "raise":
            raise CREDIT
        if self.behaviour == "swallow":
            try:
                raise CREDIT
            except Exception as e:  # noqa: BLE001
                credit.note_failure(self.credit_name, e)
                return ""
        return "ok"


def _wrapped(behaviour):
    from familiar_agent.backends.shared import watch_credit

    b = _Backend(behaviour)
    return watch_credit(_Backend._run), b


def test_an_escaping_credit_error_raises_the_alert(db):
    fn, b = _wrapped("raise")
    with pytest.raises(Exception):
        asyncio.run(fn(b))
    assert "gemini" in credit.pending()


def test_a_swallowed_credit_error_raises_the_alert_and_is_not_a_success(db):
    fn, b = _wrapped("swallow")
    assert asyncio.run(fn(b)) == ""
    assert "gemini" in credit.pending()


def test_a_call_that_goes_through_clears_the_alert(db):
    credit.note_failure("gemini", CREDIT)
    fn, b = _wrapped("ok")
    assert asyncio.run(fn(b)) == "ok"
    assert credit.pending() == {}


@pytest.mark.parametrize(
    "module,cls,name",
    [
        ("anthropic", "AnthropicBackend", "anthropic"),
        ("gemini", "GeminiBackend", "gemini"),
        ("openai_compat", "OpenAICompatibleBackend", "openai"),
        ("glm", "GLMBackend", "glm"),
        ("kimi", "KimiBackend", "kimi"),
    ],
)
def test_every_llm_backend_is_watched(module, cls, name):
    import importlib

    klass = getattr(importlib.import_module(f"familiar_agent.backends.{module}"), cls)
    assert klass.credit_name == name
    for method in ("stream_turn", "complete"):
        assert getattr(getattr(klass, method), "_watches_credit", False), (cls, method)


def test_the_swallowing_mouths_report_their_failures():
    """失敗を受け止めて空の返事を返す口は、受け止める箇所で `note_failure` を呼ぶ。"""
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "familiar_agent" / "backends"
    for f in ("anthropic.py", "gemini.py", "openai_compat.py", "glm.py", "kimi.py"):
        text = (src / f).read_text(encoding="utf-8")
        assert 'logger.warning("complete() failed: %s", e)' not in text or (
            "note_failure(self.credit_name, e)" in text
        ), f


# ── Jev ──────────────────────────────────────────────────────────────────────


def test_jev_402_raises_and_200_clears(db):
    from familiar_agent.backends.jev import JevClient

    def make(status, body):
        async def post(url, headers, payload, timeout):
            return status, body

        return JevClient(api_key="k", post=post)

    asyncio.run(make(402, {"error": "payment required"}).ask("s", {}))
    assert "jev" in credit.pending()
    asyncio.run(make(200, {"answers": {}}).ask("s", {}))
    assert "jev" not in credit.pending()
