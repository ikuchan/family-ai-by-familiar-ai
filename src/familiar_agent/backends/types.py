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
