"""「いま生きている記録」を追う仕組みを1本にした（環-g・段に）。

以前は2本あった。

- `_write_version` は版を書くたび、**同じ id を `_live_version_id` と `_chain_head_id` の
  両方**へ入れていた
- 畳む仕組みも `改訂`（`_write_version`）と `前進`（`_advance_chain`）の2つあった

`前進` が実際に発火するのは「**前の求めが閉じないまま、情動または機器で次が始まった**」
ときだけで、畳まれるのは**別々の求めの起点どうし**である。改訂（同じものの新しい状態）・
畳み込み（中身が吸われた）・解決（保留が果たされた）のどれにも当たらない。本当の目的は
「想起の枠をループ自身の記録で埋めない」ことで、**その都合のために畳むという印を使って
いた**（`設計方針_ループの語を束ねる` ②）。

求めの中は `改訂` が担うので、求めをまたいで畳む理由はない。**閉じなかった求めの起点は、
閉じなかった記録として残る**——人の発話の起点が最初から鎖の外にあって畳まれないのと、
同じ扱いになる。
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

_LOOP = Path(__file__).parent.parent / "src/familiar_agent/loop/event_loop.py"


def _code_only() -> str:
    out: list[str] = []
    with open(_LOOP, "rb") as f:
        for tok in tokenize.tokenize(io.BytesIO(f.read()).readline):
            if tok.type not in (tokenize.COMMENT, tokenize.STRING):
                out.append(tok.string)
    return " ".join(out)


def test_the_second_tracker_is_gone():
    src = _code_only()
    assert not re.search(r"\b_chain_head_id\b", src)
    assert not re.search(r"\b_advance_chain\b", src)


def test_the_advance_kind_is_no_longer_written():
    """`KIND_ADVANCE` を書く箇所が無いこと。**定数は残す**（既存の記録が参照している）。"""
    src = _code_only()
    assert "KIND_ADVANCE" not in src
    from familiar_agent.store.relations import KIND_ADVANCE

    assert KIND_ADVANCE == "前進"


def test_the_cue_survives_under_its_own_name():
    """手がかり（いま生きている記録の内容）は残る。7箇所が読んでいる。"""
    src = _code_only()
    assert not re.search(r"\b_chain_head_content\b", src)
    assert "_cue" in src


def test_all_three_starts_do_the_same_thing():
    """人の発話・情動・機器で、求めの始め方が揃っていること。

    以前は人の発話だけ `_advance_chain` を通らず、直に代入していた。同じ場面でも
    `前進` が書かれるかどうかが起点で違った。
    """
    import ast

    text = _LOOP.read_text(encoding="utf-8")
    lines = text.splitlines()
    cls = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.ClassDef) and n.name == "InformationProcessing"
    )
    starts = {}
    for m in cls.body:
        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name in (
            "begin_request",
            "_begin_affect",
            "_begin_device",
        ):
            starts[m.name] = "\n".join(lines[m.lineno - 1 : m.end_lineno])
    assert set(starts) == {"begin_request", "_begin_affect", "_begin_device"}
    for name, body in starts.items():
        assert "self._request_id = " in body, f"{name}：求めの id を置いていない"
        assert "self._note_origin(" in body, f"{name}：起点を控えていない"
        assert "self._cue = " in body, f"{name}：手がかりを置いていない"
        assert "_advance_chain" not in body, f"{name}：鎖を進めている"
