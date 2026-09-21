"""思い出すことを身体として名乗る（出-ah-ろ・2026-09-22）。

実機 16:01〜16:14、「思い出して」と 3 回頼まれても、引き方を変える 4 つの道具は 1 度も
使われなかった。道具は渡っており、主LLM も起きていた（ログで確認）。**使われなかった。**

実機の作業状態と本物の道具定義で測ると、引き直したのは 40 回中 2 回（5%）だった。
身体の定義に「思い出す」を足し、`[返事]` の行に「引き直せることをやり切ってから」を
足すと 40 回中 26 回（65%）になる。**どちらか片方だけでは 16 回中 3 回（19%）**で、
組み合わせたときだけ動く。

目・首・声・net は器官として名乗っているのに、記憶を引くことだけが道具の一覧にしか
無かった。「引き方は自分で変えられる」が身体の感覚として持てていなかった。
"""

from __future__ import annotations

from familiar_agent.loop.prompt import EVENT_SYSTEM_PROMPT
from familiar_agent.loop.reply_budget import ReplyBudget


# ── 身体に「思い出す」がある ───────────────────────────────────────────────


def _body() -> str:
    """身体の節だけを切る。規則にも `:id memory-evidence-confidence` があるので、
    全文で探すと当たってしまう（このテストを書いたとき実際に誤って通った）。"""
    return EVENT_SYSTEM_PROMPT[: EVENT_SYSTEM_PROMPT.index("(loop :id iterate")]


def test_remembering_is_named_as_a_body_part():
    assert ":id memory " in _body(), "思い出すことが身体として名乗られていない"


def test_the_four_ways_of_drawing_are_named_there():
    body = _body()
    for tool in ("recall_as", "recall_deeper", "recall_when", "recall_recent"):
        assert tool in body, f"引き方を変える道具が身体に書かれていない: {tool}"


def test_the_default_way_of_drawing_is_told():
    """いまどう引いているかを書く。固定ではないと分かるのは、既定が見えるからである。"""
    assert "自分の視点で引く" in EVENT_SYSTEM_PROMPT
    assert "引き方が合っていないだけ" in EVENT_SYSTEM_PROMPT


# ── 返事の行は「引き直したあと」と言う ────────────────────────────────────


def test_the_reply_line_says_drawing_comes_first():
    line = ReplyBudget(target=40, limit=80, max_tokens=500).line()
    assert line.startswith("[返事] 目標 40 字・80 字以内"), "字数の書き方は変えない"
    assert "引き直せることをやり切ってから" in line


def test_the_soliloquy_line_is_unchanged():
    """独り言（情動が起点）は相手への返事ではないので、引き直しの話を足さない。"""
    line = ReplyBudget(target=40, limit=80, max_tokens=500, soliloquy=True).line()
    assert line == "[独り言] 目標 40 字・80 字以内（言わなくてもよい。誰にも向けない）"
