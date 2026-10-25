"""バックエンドがやりとりする形（出-d-は）。

**最も広く使われる2つ**（`ToolCall` は17箇所、`TurnResult` は6箇所）。どのモデルの
実装からも独立しているので、いちばん下に置く。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class TurnResult:
    stop_reason: str  # "end_turn" | "tool_use"
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


def system_text(system: "str | tuple[str, ...]", sep: str = "\n\n---\n\n") -> str:
    """system を 1 本の文にする（Anthropic 以外の backend と、キャッシュを使わない主LLM）。

    組は (安定部, 可変部) か、窓の会話の中なら (安定部, 窓の会話の見出し, 行…, 可変部)（環-ar 段 D）。見出しと行は
    Anthropic のキャッシュのために分けてあるだけなので、1 本の文では 1 つの枠として改行でつなぐ。
    """
    if not isinstance(system, tuple):
        return system
    parts = system if len(system) <= 2 else (system[0], "\n".join(system[1:-1]), system[-1])
    return sep.join(p for p in parts if p)
