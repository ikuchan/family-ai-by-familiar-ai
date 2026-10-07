"""pyopenjtalk に渡す文を 1 か所の口で塊に切る（環-ad・2026-10-07 実機）。

実機 12:49・21:45・22:06 の 3 回、天気の調べものの結果が届いて 1.5〜3 秒後にアプリが落ちた。gdb の記録では、想起の語の軸
（`keyword_rules.pick_words`）が結果の文をそのまま `pyopenjtalk.run_frontend` に渡し、スタックに文（全角の URL・
「茨城県守谷市の」）がはみ出して書き込まれていた。別のプロセスで、8,002 バイトまでは通り、約 8,150 バイトで
`stack smashing detected` で落ちる。口（`core/openjtalk_safe`）で `OPENJTALK_MAX_BYTES`（4,000 バイト〔仮〕）以下に
切ってから渡し、結果をつなぐ。pyopenjtalk は中で 1 つの共有の作業物を使うので、口の中で鍵を取り、1 本ずつ呼ぶ。
"""

from __future__ import annotations

import inspect
import sys
import threading
import time
import types

import pytest

from familiar_agent.core import openjtalk_safe as oj

URLISH = "茨城県守谷市の明日の天気：ｈｔｔｐｓ：／／ｔｅｎｋｉ．ｊｐ／ｆｏｒｅｃａｓｔ／３／１１／４０２０／８２２４／ "


@pytest.fixture
def fake(monkeypatch):
    seen = {"lens": [], "inside": 0, "overlap": 0}
    lock = threading.Lock()

    def enter():
        with lock:
            if seen["inside"]:
                seen["overlap"] += 1
            seen["inside"] += 1

    def leave():
        with lock:
            seen["inside"] -= 1

    def run_frontend(text):
        enter()
        seen["lens"].append(len(text.encode("utf-8")))
        time.sleep(0.01)
        leave()
        return [{"string": text[:2], "pos": "名詞", "pos_group1": "一般"}]

    def g2p(text, kana=False):
        enter()
        seen["lens"].append(len(text.encode("utf-8")))
        time.sleep(0.01)
        leave()
        return "ヨミ"

    monkeypatch.setitem(
        sys.modules, "pyopenjtalk", types.SimpleNamespace(run_frontend=run_frontend, g2p=g2p)
    )
    return seen


def test_chunks_are_small_and_join_back():
    text = ("今日は晴れ。明日は雨！" * 50) + URLISH * 40
    parts = oj.chunks(text, max_bytes=500)
    assert "".join(parts) == text
    assert all(len(p.encode("utf-8")) <= 500 for p in parts)
    assert parts[0].endswith(("。", "！"))  # 句点のあとで切れる


def test_a_long_run_without_breaks_is_cut_at_character_boundaries():
    text = "ｔ" * 1000  # 区切りの無い全角（1 字 3 バイト）
    parts = oj.chunks(text, max_bytes=100)
    assert "".join(parts) == text
    assert all(len(p.encode("utf-8")) <= 100 for p in parts)


def test_short_text_is_one_chunk():
    assert oj.chunks("こんにちは", max_bytes=4000) == ["こんにちは"]
    assert oj.chunks("", max_bytes=4000) == []


def test_todays_text_never_reaches_pyopenjtalk_whole(fake, monkeypatch):
    monkeypatch.delenv("OPENJTALK_MAX_BYTES", raising=False)
    text = (URLISH * 200)[:2800]  # 約 8,300 バイト（今日落ちた形）
    assert len(text.encode("utf-8")) > 8150
    feats = oj.run_frontend(text)
    assert max(fake["lens"]) <= 4000 and len(fake["lens"]) >= 3
    assert len(feats) == len(fake["lens"])  # 塊ごとの結果をつなぐ


def test_g2p_joins_the_readings(fake):
    assert oj.g2p_kana("あ。" * 10, max_bytes=9) == "ヨミ" * len(fake["lens"])


def test_calls_never_overlap(fake):
    def worker():
        for _ in range(5):
            oj.run_frontend("こんにちは。")
            oj.g2p_kana("こんにちは。")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert fake["overlap"] == 0


def test_the_default_size(monkeypatch):
    monkeypatch.delenv("OPENJTALK_MAX_BYTES", raising=False)
    assert oj.max_bytes() == 4000


def test_both_callers_use_the_gate():
    from familiar_agent.core import keyword_rules, reading

    assert "openjtalk_safe" in inspect.getsource(keyword_rules.pick_words)
    assert "pyopenjtalk.run_frontend" not in inspect.getsource(keyword_rules.pick_words)
    assert "openjtalk_safe" in inspect.getsource(reading._g2p)
    assert "pyopenjtalk.g2p" not in inspect.getsource(reading._g2p)
