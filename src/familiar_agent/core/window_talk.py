"""窓の会話（環-ar 段 A・2026-10-25・本人の決定）。純関数と器。

窓が開いてから閉じるまでの会話を、ひとつの会話として 1 行ずつ持つ。Jev・軽量LLM・主LLM に渡すのは、数と時間で切った
直近（主 6 往復・軽量と Jev 3 往復・5 分まで）ではなく、この窓の会話の全部である。窓の外では、いまどおりの直近に戻る。

**行は出来事が起きたその場で足し、書いた行は書き換えない。** 主LLM には 1 行ずつを別のまとまりにして渡し、最後の行に
キャッシュの印を置く。前の行が 1 字でも変わると、そこから後ろのキャッシュが外れる。DB から引き直さないのも同じ理由で、
後から話者を付け替えても、前の行の名前は変わらない（直しは、その言葉が新しい行として載る）。

調べた結果は短い 1 行（`RESULT_MAX_CHARS`）で、全文はいまどおり作業状態の W に置く。器はメモリ上にだけ置き、再起動で
消えてよい（窓も消える）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

#: 調べた結果の 1 行に載せる字数（2026-10-25 本人の決定）。
RESULT_MAX_CHARS = 80


def _one_line(text: str) -> str:
    return " ".join(str(text or "").split())


def result_text(query: str, result: str) -> str:
    """調べた結果を、窓の会話の 1 行の本文にする（「調べた（語）：結果」・結果は `RESULT_MAX_CHARS` 字まで）。"""
    body = _one_line(result)
    if len(body) > RESULT_MAX_CHARS:
        body = body[:RESULT_MAX_CHARS] + "…"
    return f"調べた（{_one_line(query)[:RESULT_MAX_CHARS]}）：{body}"


@dataclass
class WindowTalk:
    """1 つの窓の会話。`session` は `WakeWindow.session`、`opened_at` は窓が開いた時刻（その場の時計）。"""

    session: int
    opened_at: datetime
    lines: list[str] = field(default_factory=list)
    #: この窓の、いま生きている最終版の id（段 E：窓の中の次の最終版が改訂で畳む）。
    final_id: str = ""

    def add(self, at: datetime, who: str, text: str) -> None:
        """1 行足す。前の行には触らない。"""
        body = _one_line(text)
        if body:
            self.lines.append(f"- {at:%H:%M:%S} {who}：{body}")

    def header(self) -> str:
        return f"[窓の会話（{self.opened_at:%H:%M:%S} から・古い順・全部）]"

    def blocks(self) -> "tuple[str, ...]":
        """見出しと、1 行ずつ（主LLM へ別のまとまりとして渡す）。行が無ければ空。"""
        if not self.lines:
            return ()
        return (self.header(), *self.lines)

    def render(self) -> str:
        """1 つの文（Jev・軽量LLM へ渡す）。行が無ければ空。"""
        return "\n".join(self.blocks())
