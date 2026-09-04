"""軽量LLM へ「形のある答え」を求める口。

**モデルを替えたとき壊れるのは、ここだった。** `complete()` は文字列を返すだけなので、
呼び出し側が7箇所でそれぞれ解釈していた——数値を正規表現で拾う、集合に含まれるか見る、
`yes` で始まるか見る、JSON として読む。同じ問題を別々に解いており、外れたときの扱いも
割れていた（記録して倒すもの／黙って既定にするもの）。

**返すのは「取れたか、取れなかったか」だけである。** 取れなければ `None` を返し、既定へ
落とすのは呼び出し側が決める。測れなかったものを口が勝手に埋めない（050 で PAD について
決めたのと同じ形）。

**取れなかったことを数える。** どのモデルがどの形をどれだけ外すかが分かれば、それがモデル
選定の判断材料になる。以前は `.startswith("yes")` が「はい、同じです」を黙って偽にしていた。

**答えの言語を問わない。** 「短く `yes` か `no` で」と指示しても、日本語で応じるモデルがある。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .helpers import strip_code_fence

logger = logging.getLogger(__name__)

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

#: 肯定・否定と読める語。モデルは指示した言語で答えるとは限らない。
_YES = ("yes", "true", "same", "はい", "同じ", "そう")
_NO = ("no", "false", "different", "いいえ", "違", "異な")

#: 「どれも当てはまらない」と答えるときの語（部分集合を求めたとき）。
_NONE_WORDS = ("none", "なし", "無し", "ありません")


async def _ask(backend, prompt: str, max_tokens: int, want: str) -> str | None:
    """軽量LLM を呼び、生の返事を返す。落ちたら `None`（例外は投げない）。"""
    try:
        raw = await backend.complete(prompt, max_tokens=max_tokens)
    except Exception as e:  # noqa: BLE001
        logger.warning("形のある答えを求めたが呼び出しに失敗した（%s）: %s", want, e)
        return None
    return str(raw or "")


def _unusable(want: str, raw: str) -> None:
    """形が取れなかったことを残す。**どのモデルが何を外すかを測るための一行。**"""
    logger.warning("形のある答えを読めなかった（%s）: %.120r", want, raw)


async def ask_numbers(
    backend, prompt: str, *, count: int, lo: float = 0.0, hi: float = 1.0,
    max_tokens: int = 20,
) -> "tuple[float, ...] | None":
    """数値を `count` 個、出てきた順に読む。範囲へ丸める。足りなければ `None`。"""
    raw = await _ask(backend, prompt, max_tokens, f"数値{count}個")
    if raw is None:
        return None
    nums = _NUMBER.findall(raw)
    if len(nums) < count:
        _unusable(f"数値{count}個", raw)
        return None
    try:
        return tuple(min(hi, max(lo, float(n))) for n in nums[:count])
    except ValueError:
        _unusable(f"数値{count}個", raw)
        return None


async def ask_choice(
    backend, prompt: str, *, choices: "set[str] | frozenset[str]", max_tokens: int = 10,
) -> str | None:
    """選択肢から1つ選ばせる。**2つ以上あてはまるなら選べていない**ので `None`。

    短く答えろと指示しても文で返すことがあるので、含まれていれば拾う。
    """
    raw = await _ask(backend, prompt, max_tokens, "選択肢1つ")
    if raw is None:
        return None
    low = raw.lower()
    hit = sorted(c for c in choices if c.lower() in low)
    if len(hit) != 1:
        _unusable("選択肢1つ", raw)
        return None
    return hit[0]


async def ask_yes_no(backend, prompt: str, *, max_tokens: int = 5) -> bool | None:
    """はい／いいえ。**どちらとも読めなければ `None`**（黙って偽にしない）。"""
    raw = await _ask(backend, prompt, max_tokens, "はい／いいえ")
    if raw is None:
        return None
    low = raw.lower()
    yes = any(w in low for w in _YES)
    no = any(w in low for w in _NO)
    if yes == no:          # 両方あるか、どちらも無い
        _unusable("はい／いいえ", raw)
        return None
    return yes


async def ask_subset(
    backend, prompt: str, *, choices: "set[str] | frozenset[str]", max_tokens: int = 32,
) -> "frozenset[str] | None":
    """選択肢のうち当てはまるものを列挙させる。

    **「どれも無い」は取れた答えであって失敗ではない**ので、空集合を返す。何も読めない
    ときだけ `None`。
    """
    raw = await _ask(backend, prompt, max_tokens, "選択肢の部分集合")
    if raw is None:
        return None
    low = raw.lower()
    hit = frozenset(c for c in choices if c.lower() in low)
    if hit:
        return hit
    if any(w in low for w in _NONE_WORDS):
        return frozenset()
    _unusable("選択肢の部分集合", raw)
    return None


def read_json(raw: str) -> "dict[str, Any] | None":
    """文字列から JSON の物体を読む。**コードフェンスは剥がす。** 読めなければ `None`。

    実機の VLM は同じ画像・同じプロンプトでも回ごとに ```` ```json ```` で包んだり包まな
    かったりする（`core.helpers.strip_code_fence` の実測）。配列だけを返してきたときは、
    物体を求めているので取れなかったものとして扱う。

    **記録はしない。** 呼ぶ側が生の返事を持っているので、どう残すかは呼ぶ側が決める。
    """
    try:
        loaded = json.loads(strip_code_fence(raw))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


async def ask_json(backend, prompt: str, *, max_tokens: int = 512) -> "dict[str, Any] | None":
    """JSON の物体を求める。読めなければ `None`（読めなかったことは記録する）。

    生の返事を見て独自に記録したい呼び出し側は、`complete` を自分で呼んで `read_json` を
    使う（`scene.py` は「空だったのか説明文だったのか」で対応が変わる）。
    """
    raw = await _ask(backend, prompt, max_tokens, "JSON")
    if raw is None:
        return None
    data = read_json(raw)
    if data is None:
        _unusable("JSON", raw)
    return data
