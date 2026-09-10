"""反復の文脈の文面を、生成器へ出す（環-e-に・に-5-は）。

`_iterate` は主LLM へ渡す `[反復] …` の文面を、その場で 13 行かけて組み立てていた。
**ループの可変状態を1つも読まない**——数（何反復目・上限・考えた回数）と真偽（上限に
達したか）だけで決まる文字列である。

`loop/generator.py` は に-1 でその置き場として作った（`_present_ctx`・`_pi_ctx` と同じ
family）。**挙動は変えない。**

`build_event_system_prompt` の呼び出しそのものは動かさない。あれは `loop/prompt.py` への
薄い呼び出しで、包み直しても 9 個の引数を素通しするだけになる（に-5-い で剥がした形が
戻る）。
"""

from __future__ import annotations

import inspect

from familiar_agent.loop.event_loop import InformationProcessing
from familiar_agent.loop.generator import _iter_ctx


def _ctx(**kw) -> str:
    base = dict(chain=1, max_chain=5, thinking_round=1, capped=False)
    base.update(kw)
    return _iter_ctx(**base)


# ── 文面 ───────────────────────────────────────────────────────────────────


def test_it_carries_the_iteration_and_the_cap():
    assert "[反復] 2/5" in _ctx(chain=2)


def test_it_carries_how_many_times_it_thought():
    """反復の数は主LLM の返りで 0 へ戻るので、求めの長さを表すのはこちら（環-h ⑥-2）。"""
    assert "3 回目" in _ctx(thinking_round=3)


def test_at_the_cap_it_says_so():
    """上限では、黙って手持ちで繕わず「調べきれなかった」と断ってから答えさせる。"""
    assert "これ以上は調べられない" in _ctx(capped=True)


def test_below_the_cap_it_does_not():
    assert "これ以上は調べられない" not in _ctx(capped=False)


# ── 呼び手 ─────────────────────────────────────────────────────────────────


def test_the_iteration_does_not_compose_the_text_anymore():
    src = inspect.getsource(InformationProcessing._iterate)
    assert "[反復]" not in src, "反復が文面を組んでいる"
    assert "これ以上は調べられない" not in src


def test_the_iteration_got_shorter():
    """に-5-は で 212 → 202 行になった。

    に-5-に-2 で **205 行へ 3 行戻っている**。W を核（`loop/workspace.py`）へ出したとき、
    委ねる呼び出し（`workspace.recall(...)`）が 1 行に収まらず 3 行に折れたためである。
    `event_loop.py` 全体は 1,905 → 1,737 行なので、**核を出す代わりに殻の呼び口が伸びた**
    という取引である。数は隠さず、動いたら書き換える。

    出-h-ろ で **215 行へ 10 行増えた**。`light` で閉じる反復にも申告の口を通したためで
    （`_declare_light_memory_use`）、**記憶が育つ経路を1本に保つ**ための増分である。
    中身は殻の呼び口1つ（背景で投げるだけ）で、判断は核（`workspace.ask_verdicts`）にある。
    """
    src = inspect.getsource(InformationProcessing._iterate)
    assert len(src.split("\n")) <= 215, "薄くなっていない"
