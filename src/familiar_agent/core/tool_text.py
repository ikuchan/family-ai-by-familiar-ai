"""道具呼び出しが**文として**返ってきたときに拾う（2026-09-13・2026-09-15）。

主LLM（Sonnet 5）が `<invoke name="recall"><parameter name="query">…</parameter></invoke>` を
tool_use ブロックでなく本文として返した（約 50 回に 1 回）。プロンプトにこの形は無く、
モデル側の取りこぼしだが、ループは素テキストとして扱い、画面と O にそのまま出してしまった。
形が読めるなら呼び出しとして拾う。読めなければ何もしない（呼び手が素テキストとして扱う）。

2 つ目の形：**関数呼び出し風** `say(text="…", memory_verdicts=[…])`（2026-09-15・パジュへのメモの
初回で、答えは作れていたのに「道具なし」と見て沈黙になった）。本文の**先頭**が `名前(` で始まり、
`key=値` の並び（値は JSON か単引用符の文字列）で閉じているときだけ拾う。
"""

from __future__ import annotations

import json
import re

from ..backends.types import ToolCall

_INVOKE = re.compile(r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>', re.S)
_PARAM = re.compile(r'<parameter\s+name="([^"]+)"\s*>(.*?)</parameter>', re.S)


def _coerce(value: str):
    v = value.strip()
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d+\.\d+", v):
        return float(v)
    if v in ("true", "false"):
        return v == "true"
    return v


_CALL_HEAD = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\(", re.S)
_KEY = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*")
_decoder = json.JSONDecoder()


def _call_style(text: str) -> "ToolCall | None":
    """`名前(key=値, …)` を 1 つ読む。全体が読めなければ None（本文のまま）。"""
    m = _CALL_HEAD.match(text or "")
    if not m or not text.rstrip().endswith(")"):
        return None
    name = m.group(1)
    body = text.rstrip()[m.end() : -1]
    pos = 0
    params: dict = {}
    while pos < len(body):
        km = _KEY.match(body, pos)
        if not km:
            return None
        key = km.group(1)
        pos = km.end()
        try:
            if pos < len(body) and body[pos] == "'":
                end = body.index("'", pos + 1)
                value, pos = body[pos + 1 : end], end + 1
            else:
                value, pos = _decoder.raw_decode(body, pos)
        except (ValueError, json.JSONDecodeError):
            return None
        params[key] = value
        rest = body[pos:].lstrip()
        if not rest:
            break
        if not rest.startswith(","):
            return None
        pos = len(body) - len(rest) + 1
    return ToolCall(id="text-0", name=name, input=params)


def tool_calls_from_text(text: str) -> list[ToolCall]:
    """本文に書かれた `<invoke>` か、先頭の関数呼び出し風を `ToolCall` の並びにする。無ければ空。"""
    calls: list[ToolCall] = []
    for i, m in enumerate(_INVOKE.finditer(text or "")):
        name, body = m.group(1).strip(), m.group(2)
        params = {k.strip(): _coerce(v) for k, v in _PARAM.findall(body)}
        calls.append(ToolCall(id=f"text-{i}", name=name, input=params))
    if not calls:
        one = _call_style(text)
        if one is not None:
            calls.append(one)
    return calls
