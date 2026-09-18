"""確認待ち（出-y・2026-09-18・`設計方針_タイマー` v0.9 §10）。

掛ける前の確認（「3 分のタイマーね、いい？」）は、**求めではなく機械の状態**が持つ。以前は道具の
引数 `confirmed` で表し、「いい」と言われた LLM が同じ引数に `confirmed=true` を足して呼び直す
約束だったが、主LLM が聞く前に `confirmed=true` を付けて掛けた（実機 15:42・id=13）。LLM に
「掛け直す」判断を渡す限り、この穴は閉じない。

道具が「確かめて」と判定したら預かり（`PendingConfirm`）を置き、返りは確認文だけ。次の求めで
預かりが生きていれば（`CONFIRM_TTL_SEC`）W の最上部に `[確認待ち]`、調停の候補に `confirm`／
`decline`。`confirm` は機械が預かった入力で道具を呼ぶ（`call(..., confirmed=True)`・キーワード
引数で、LLM の入力からは届かない）。`decline` は捨てる。別の `set_timer` が来れば上書き。
状態はメモリ上（`agent._pending_confirm`）。再起動で消えてよい（5 分〔仮〕の預かり）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PendingConfirm:
    action: str  # 掛ける道具（`set_timer`・`set_alarm`）
    tool_input: dict = field(default_factory=dict)  # 預かった入力（そのまま呼び直す）
    asked_at: float = 0.0  # 聞いた時刻（`time.time()`）
    text: str = ""  # 本人に聞いた文
    what: str = ""  # 何を（枠の見出し・例「タイマーを掛ける「パスタ」（3 分）」）


def alive(pc: "PendingConfirm | None", *, now: float, ttl: float) -> bool:
    """預かりが生きているか（聞いてから `ttl` 秒以内）。"""
    return pc is not None and (now - pc.asked_at) <= ttl


def frame(pc: PendingConfirm) -> str:
    """W の最上部に載せる `[確認待ち]` の枠。答えが「いい」なら `confirm`、「やめて」なら `decline`。"""
    return f"[確認待ち] {pc.what}：「{pc.text}」——「いい」「うん」なら confirm、「やめて」「いらない」なら decline"


#: 調停の候補と主LLM の道具に同じ 2 本を載せる（預かりが生きているあいだだけ）。
TOOL_DEFS: list[dict] = [
    {
        "name": "confirm",
        "description": "[確認待ち] の問いに「いい」「うん」「お願い」と答えた。預かった入力で掛ける。引数は無い。",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "decline",
        "description": "[確認待ち] の問いに「やめて」「いらない」「今はいい」と答えた。預かりを捨てる。引数は無い。",
        "input_schema": {"type": "object", "properties": {}},
    },
]
