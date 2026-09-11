"""**記憶への線を1本にする**（環-e-い・書き込み側）。

`設計図` ③-2 は「口は4つだけ（IIF・DIF・AIF・OIF）。この4つ以外に、コンポーネントどうしが
直接つながる線は置かない」と定める。OIF は**記憶ストア O との唯一の出入り口**である。

ところが OIF は**口が建っただけで誰も通っていなかった**（`OIF(` の生成が本番コードに0件）。
記憶へは各所が `agent._memory` を直に触っていた。

**まず書き込みを通す。** `write`／`append`／`supersede` は器が変わらない（id の文字列を
返すだけ）ので、挙動を変えずに移せる。しかも `OIF.write` は書き手を必須で受け取るので、
通した箇所すべてで**誰の記録かが明示される**。

**残すもの。** 関係（`record_exchange`・`record_succession`・`recent_exchanges`・
`latest_exchange_origin`）は OIF に口が無い——関係は 058〜060 で入った機構で、2026-08-01 に
設計した OIF が追いついていない。想起は器が違う（`Recalled` 対 辞書の並び）。`loop/rest.py`
は `direction='内省'` に `kind='observation'` を渡しており、OIF の表（`内省`→`self_model`）と
食い違う——どちらが正かは記-a（REST 内省）で決める。
"""

from __future__ import annotations

import ast
import inspect
import pathlib

_LOOP = pathlib.Path(__file__).resolve().parent.parent / "src" / "familiar_agent" / "loop"


def _direct_memory_calls(name: str) -> list[tuple[int, str]]:
    """`event_loop.py` が `agent._memory.<何か>` を直に呼んでいる箇所。"""
    f = _LOOP / "event_loop.py"
    tree = ast.parse(f.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if getattr(fn, "attr", None) != name:
            continue
        owner = ast.unparse(fn.value) if hasattr(fn, "value") else ""
        if owner.endswith("_memory"):
            out.append((node.lineno, owner))
    return out


def test_the_loop_does_not_write_to_memory_directly():
    """書き込みは口を通る。`save_async_with_id`・`mark_superseded`・`note_lookup_started`。"""
    left = {
        n: _direct_memory_calls(n)
        for n in ("save_async_with_id", "mark_superseded", "note_lookup_started")
    }
    left = {k: v for k, v in left.items() if v}
    assert not left, f"記憶へ直に書いている: {left}"


def test_the_loop_uses_the_mouth():
    """**反証側。** 直の呼び出しが消えただけで口を通っていなければ、何も守っていない。"""
    src = (_LOOP / "event_loop.py").read_text(encoding="utf-8")
    assert src.count("_oif.write(") == 6
    assert src.count("_oif.supersede(") == 2
    assert src.count("_oif.append(") == 1


def test_the_agent_holds_one_mouth():
    from familiar_agent.agent import EmbodiedAgent

    src = inspect.getsource(EmbodiedAgent.__init__)
    assert "self._oif = OIF(" in src


def test_what_is_left_is_named():
    """**残したものを黙って落とさない。** 書き込み・関係・埋め込み・想起は通した。

    残るのは**公開面を絞ること**（`ObservationMemory` は 72 種）と、`loop/rest.py` の
    `kind` の食い違い（記-a で決める）である。
    """
    w = (_LOOP / "workspace.py").read_text(encoding="utf-8")
    assert "mem.recall_async(" not in w and "mem.format_for_context(" not in w
    assert "oif.recall(" in w
