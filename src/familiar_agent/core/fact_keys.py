"""記録の本文から事実を抜く頼みと、その返りの読み方と、鍵の文（記-ad・2026-11-20）。

論文 AtlasKV の KG2KV の考え方：事実を「主語・関係・答え」の 3 つ組にし、片側を隠して問いに近い文にする。
- 答えを隠す向き：「〈主語〉の〈関係〉」（例：町内会の夏祭りの日）
- 主語を隠す向き：「〈関係〉が「〈答え〉」であるもの」
言い換えた問い（「夏祭りっていつ？」）が、短く書き方の揺れた記録の本文より、鍵の文に近くなる。

ここは純関数だけ。LLM を呼ぶのと書くのは `loop/rest_facts.py`。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .helpers import strip_code_fence

MAX_CHARS = 1500  # 1 件の本文を頼みに載せる上限（長い自己エピソードも頭の 1500 字で足りる）
MAX_FACTS = 8  # 1 件から抜く事実の上限
_ARRAY_START = re.compile(r"\[\s*[\{\]]")  # JSON の配列の始まり（`[{` か `[]`）

PROMPT = """次の記録それぞれから、あとで家族の質問に答えるのに役立つ事実を、「主語・関係・答え」の 3 つ組で抜き出してください。

- 関係は名詞にする。例：（主語「町内会」、関係「夏祭りの日」、答え「8月3日」）、（主語「エアコン」、関係「止まった理由」、答え「フィルターの詰まり」）。
- 表記は一つにそろえ、略語は開く（例：健診→健康診断）。言い間違いや聞き間違いがあれば、正しい言い方も答えに書く。
- 日時、曜日、時刻、数、値段、場所、人、物、理由、結果、好み、予定など、具体的なものを優先する。
- 気持ちの描写だけ、挨拶だけ、パジュが見回った・調べたという動作の報告だけの記録からは、何も抜かない。
- 1 件あたり最大 8 個。抜くものが無ければ空にする。
- 主語が書かれていなければ、記録から分かる範囲で補う（パジュ、パパ、ママ、子ども、家族など）。

出力は JSON だけにする（説明は書かない）：
[{"i": 記録の番号, "facts": [{"s": "主語", "r": "関係", "a": "答え"}]}]

記録：
"""


@dataclass(frozen=True)
class Fact:
    """記録から抜いた事実 1 つ。"""

    s: str
    r: str
    a: str

    def keys(self) -> "list[tuple[str, str]]":
        """(向き, 鍵の文) を 2 つ。向きは表 `obs_fact_keys.side` の値。主語が空なら答えを隠す向きは関係だけ。"""
        return [
            ("答え", f"{self.s}の{self.r}" if self.s else self.r),
            ("主語", f"{self.r}が「{self.a}」であるもの"),
        ]


def build_prompt(texts: "list[str]") -> str:
    """記録の本文を番号つきで並べた頼み。改行は空白にする（番号の行を崩さない）。"""
    body = "\n".join(f"[{i}] " + t[:MAX_CHARS].replace("\n", " ") for i, t in enumerate(texts))
    return PROMPT + body


def parse(raw: "str | None", n: int) -> "list[list[Fact] | None] | None":
    """返りを記録ごとの事実の並びにする（長さ n）。JSON の配列が読めなければ None（その束は抜いていない扱い）。

    - 空の配列は、どの記録も事実 0 個（頼みは「抜くものが無ければ空」と言っている）。
    - 返りに番号の無い記録は None（印を付けず、次に抜き直す）。項があるのにどの番号も当たらなければ全体が None。
    - 番号は整数か数字の文字（真偽値は外す）。範囲の外の番号、形の崩れた項、関係か答えの無い事実は捨てる。
    - 配列は `[{` か `[]` の形の始まりから読む（前置きの文の `[` で崩れない）。
    """
    text = strip_code_fence(raw or "")
    m = _ARRAY_START.search(text)
    hi = text.rfind("]")
    if m is None or hi < m.start():
        return None
    try:
        data = json.loads(text[m.start() : hi + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, list):
        return None
    if not data:
        return [[] for _ in range(n)]
    out: "list[list[Fact] | None]" = [None] * n
    for item in data:
        if not isinstance(item, dict):
            continue
        i = item.get("i")
        if isinstance(i, str) and i.strip().isdigit():
            i = int(i)
        facts = item.get("facts")
        if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n:
            continue
        if not isinstance(facts, list):
            continue
        prev = out[i]
        got: "list[Fact]" = prev if prev is not None else []
        for f in facts:
            if not isinstance(f, dict):
                continue
            r, a = str(f.get("r") or "").strip(), str(f.get("a") or "").strip()
            if r and a and len(got) < MAX_FACTS:
                got.append(Fact(str(f.get("s") or "").strip(), r, a))
        out[i] = got
    return out if any(x is not None for x in out) else None
