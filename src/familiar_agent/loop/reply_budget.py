"""返事の予算（reply budget）——長さと `max_tokens` を求めごとに規則で決める（出-k-ろ）。

規則「say() は 1〜2 文」だけでは、実測で 60〜195 字（3〜5 文）が通った（`根拠台帳` §32）。
長さは**機械で決めて数字で渡す**。目標と上限の両方を渡すのは、上限だけだとそこまで膨らむ
ためである。値は `課題5` G 章（2026-09-12 決定）。

`max_tokens` は**出力だけ**の上限で、返事の文字・`say` の JSON と記憶の申告・内部の思考を
含む（実 API で確認：思考が要る問いで 150 は思考だけで使い切り本文が空になった）。だから
上限字数だけで決めず、申告分と思考分を足す。思考しない問いでは余るだけで害は無い。
"""

from __future__ import annotations

from dataclasses import dataclass

#: 場面ごとの（目標, 上限）字数。
_DEFAULT = (40, 80)  # low／medium・調べていない
_HIGH = (80, 160)  # 人が「よく考えて」と明示したとき
_RESEARCHED = (160, 240)  # この求めでネット調査をした（effort に依らず優先）
_SOLILOQUY = (20, 40)  # 情動が起点（独り言・情-e）

#: 思考分（トークン）。effort ごとに、考えるかもしれない分を積む。
_THINKING = {"low": 256, "medium": 1024, "high": 2048}

#: 申告分（トークン）：W の 1 件あたり（12 桁の id ＋ 判定語 ＋ JSON の骨）と、`say` の骨。
_PER_VERDICT = 20
_SAY_OVERHEAD = 60

#: 日本語 1 字あたりのトークンの見積もり（Claude は 1〜1.5）。上限の 2 倍を返事の枠にする。
_TOKENS_PER_CHAR = 2


@dataclass(frozen=True)
class ReplyBudget:
    target: int  # 目標字数（主LLM に渡す）
    limit: int  # 上限字数（主LLM に渡す）
    max_tokens: int  # API の出力上限

    soliloquy: bool = False  # 情動が起点（独り言）。誰にも向けない・言わなくてもよい

    def line(self) -> str:
        """主LLM へ渡す 1 行。"""
        if self.soliloquy:
            return (
                f"[独り言] 目標 {self.target} 字・{self.limit} 字以内"
                "（言わなくてもよい。誰にも向けない）"
            )
        # 字数だけを置くと「いま返事を書く」の合図になり、引き直せるのに諦める（出-ah-ろ）。
        # **身体の `:id memory` と対で効く**——片方だけでは 16 回中 3 回、両方で 40 回中 26 回。
        # 独り言（情動が起点）には足さない。相手への返事ではないので、引き直す話が要らない。
        return (
            f"[返事] 目標 {self.target} 字・{self.limit} 字以内"
            "（返事を書くのは、引き直せることをやり切ってから。"
            "思い出せていないなら、この行はまだ自分に向いていない）"
        )


def decide(*, effort: str, researched: bool, w_count: int, origin: str = "発話") -> ReplyBudget:
    """長さと `max_tokens` を決める。

    - `effort`：調停が決めた思考の深さ（low／medium／high。知らない値は low）
    - `researched`：この求めで `search_deferred`／`fetch_deferred` を投げたか
    - `w_count`：W に載った記憶の件数（申告 1 件ずつぶんのトークン）
    """
    effort = effort if effort in _THINKING else "low"
    if origin == "情動":
        target, limit = _SOLILOQUY
    elif researched:
        target, limit = _RESEARCHED
    elif effort == "high":
        target, limit = _HIGH
    else:
        target, limit = _DEFAULT
    max_tokens = (
        limit * _TOKENS_PER_CHAR + (w_count * _PER_VERDICT + _SAY_OVERHEAD) + _THINKING[effort]
    )
    return ReplyBudget(
        target=target, limit=limit, max_tokens=max_tokens, soliloquy=origin == "情動"
    )
