"""道具呼び出しが**文として**返ってきたときに拾う（2026-09-13）。

主LLM（Sonnet 5）が `<invoke name="recall"><parameter name="query">…</parameter></invoke>` を
tool_use ブロックでなく本文として返した（約 50 回に 1 回）。プロンプトにこの形は無く、
モデル側の取りこぼしだが、ループは素テキストとして扱い、画面と O にそのまま出してしまった。
形が読めるなら呼び出しとして拾う。読めなければ何もしない（呼び手が素テキストとして扱う）。
"""

from __future__ import annotations

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


def tool_calls_from_text(text: str) -> list[ToolCall]:
    """本文に書かれた `<invoke>` を `ToolCall` の並びにする。無ければ空。"""
    calls: list[ToolCall] = []
    for i, m in enumerate(_INVOKE.finditer(text or "")):
        name, body = m.group(1).strip(), m.group(2)
        params = {k.strip(): _coerce(v) for k, v in _PARAM.findall(body)}
        calls.append(ToolCall(id=f"text-{i}", name=name, input=params))
    return calls
