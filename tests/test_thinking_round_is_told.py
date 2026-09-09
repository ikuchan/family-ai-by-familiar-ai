"""何回目に考えているかを、両方へ伝える（環-h）。

主LLM を投げっぱなしにし、返りで反復をリセットしたとき、**「この求めで何回目か」の手がかりが
3つとも同時に消えた**。

- 主LLM のプロンプトの `[反復] N/M` は、決める反復だけを数えてリセットされるので常に小さい
- 調停（軽量LLM）へは回数そのものを渡していない
- `capped`（上限に達したか）も、リセットするので立たない

この機体の書き方は「**抑止で黙らせるのではなく、判断できる材料を渡して解く**」である
（`_say_filler` のコメント）。機械の歯止めは置かず、**回数を材料として渡す**。

数えるのは器（`_lookups`）の `action="主LLM"` で、`_finish` と打ち切りで空になるので
**求めごとの回数**になる。
"""

from __future__ import annotations

from familiar_agent.loop.arbiter import ARBITER_PROMPT, arbitrate
from familiar_agent.loop.event_loop import InformationProcessing, Lookup
from familiar_agent.loop.request import Request


def _ip(rounds: int = 0):
    ip = InformationProcessing.__new__(InformationProcessing)
    # `__new__` は `__init__` を通らないので、求めの器は自分で置く（に-5-に-1）。
    ip._req = Request()
    ip._req.lookups = [
        Lookup(index=i + 1, action="主LLM", query=f"主LLM{i + 1}", generation=0, result="済")
        for i in range(rounds)
    ]
    return ip


# ── 数える ─────────────────────────────────────────────────────────────────


def test_the_first_call_is_the_first_round():
    assert _ip(0)._thinking_round == 1


def test_each_past_call_raises_the_round():
    assert _ip(2)._thinking_round == 3


def test_lookups_other_than_the_main_llm_are_not_counted():
    ip = _ip(1)
    ip._req.lookups.append(Lookup(index=9, action="recall", query="q", generation=0))
    assert ip._thinking_round == 2


# ── 主LLM へ伝える ─────────────────────────────────────────────────────────


def test_the_main_llm_is_told_which_round_it_is():
    import inspect

    src = inspect.getsource(InformationProcessing._iterate)
    assert "_thinking_round" in src, "反復の文脈に回数が入っていない"


# ── 調停へ伝える ───────────────────────────────────────────────────────────


def test_the_arbiter_takes_the_round():
    import inspect

    assert "thinking_round" in inspect.signature(arbitrate).parameters


def test_the_arbiter_prompt_has_a_slot_for_it():
    assert "{thinking_note}" in ARBITER_PROMPT
