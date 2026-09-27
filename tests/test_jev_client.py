"""Jev（判断専用モデル）を呼ぶ口（出-au 段 4-1・2026-09-27・`設計方針_判定の段` §2.2）。

`POST https://api.typesafe.ai/v1/systemone`（`Authorization: Bearer <鍵>`・`model`・`state`・`questions`）。
答えは質問ごとに Choice（選んだもの・全選択肢の確率・確信度）／Score／Noul で返る（`docs.typesafe.ai`）。
判定の段では、**失敗しても例外にしない**。鍵が無い・時間切れ・429／529・形の誤りは「失敗」を返し、呼び手が
判定ごとの既定へ倒す。入っている `aiohttp` で呼ぶ（依存を増やさない）。
"""

from __future__ import annotations

import asyncio

from familiar_agent.backends.jev import JevClient, choice, noul, score


class _Fake:
    def __init__(self, status=200, body=None, exc=None, delay=0.0):
        self.status, self.body, self.exc, self.delay = status, body, exc, delay
        self.calls: list = []

    async def __call__(self, url, headers, payload, timeout):
        self.calls.append((url, headers, payload, timeout))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return self.status, self.body


_OK = {
    "model": "jev-1.13.0",
    "answers": {
        "to_whom": {
            "type": "choice",
            "choice": "family",
            "probabilities": {"paju": 0.2, "family": 0.7, "unknown": 0.1},
            "confidence": 0.78,
        }
    },
    "usage": {"input_tokens": 120, "output_tokens": 9},
}


def _client(fake, key="k-test", timeout=2.0):
    return JevClient(api_key=key, model="jev-latest", timeout=timeout, post=fake)


def test_it_sends_the_documented_request():
    fake = _Fake(body=_OK)
    q = {"to_whom": choice("誰に向けた言葉か", {"paju": "パジュ宛て", "family": "家族どうし"})}
    asyncio.run(_client(fake).ask("状況", q))
    url, headers, payload, timeout = fake.calls[0]
    assert url == "https://api.typesafe.ai/v1/systemone"
    assert headers["Authorization"] == "Bearer k-test"
    assert payload == {
        "state": "状況",
        "model": "jev-latest",
        "questions": {
            "to_whom": {
                "type": "choice",
                "instructions": "誰に向けた言葉か",
                "criteria": {"paju": "パジュ宛て", "family": "家族どうし"},
            }
        },
    }
    assert timeout == 2.0


def test_it_reads_the_answer_with_probabilities_and_confidence():
    got = asyncio.run(_client(_Fake(body=_OK)).ask("状況", {"to_whom": noul("x")}))
    assert got.ok and got.model == "jev-1.13.0"
    a = got.answers["to_whom"]
    assert a["choice"] == "family" and a["confidence"] == 0.78
    assert a["probabilities"]["family"] == 0.7
    assert got.seconds >= 0.0


def test_question_builders_match_the_three_types():
    assert score("強さ", ["弱い", "強い"]) == {
        "type": "score",
        "instructions": "強さ",
        "criteria": ["弱い", "強い"],
    }
    assert noul("急ぎか") == {"type": "noul", "instructions": "急ぎか"}


def test_without_a_key_it_does_not_call():
    fake = _Fake(body=_OK)
    got = asyncio.run(_client(fake, key="").ask("状況", {"q": noul("x")}))
    assert not got.ok and got.error == "鍵が無い" and fake.calls == []


def test_failures_are_returned_not_raised():
    for fake, why in (
        (_Fake(status=429, body={}), "429"),
        (_Fake(status=529, body={}), "529"),
        (_Fake(status=401, body={}), "401"),
        (_Fake(exc=ConnectionError("down")), "ConnectionError"),
        (_Fake(status=200, body={"nope": 1}), "形"),
    ):
        got = asyncio.run(_client(fake).ask("状況", {"q": noul("x")}))
        assert not got.ok and why in got.error, (why, got.error)


def test_a_slow_answer_times_out_as_a_failure():
    got = asyncio.run(_client(_Fake(body=_OK, delay=0.5), timeout=0.05).ask("s", {"q": noul("x")}))
    assert not got.ok and "時間切れ" in got.error


def test_the_key_and_model_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "k-env")
    monkeypatch.setenv("JEV_MODEL", "jev-1.13")
    c = JevClient.from_env()
    assert c.available and c.model == "jev-1.13"
    monkeypatch.delenv("JEV_API_KEY")
    monkeypatch.delenv("JEV_MODEL")
    c = JevClient.from_env()
    assert not c.available and c.model == "jev-latest"
